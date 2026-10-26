"""Windows video-capture device enumeration via SetupAPI (KSCATEGORY_VIDEO_CAMERA).

OpenCV CAP_MSMF indexes the same capture endpoints MFEnumDeviceSources returns;
SetupAPI's present video-camera interfaces are that set (not the broader PnP
Camera class, which can include non-capture siblings like IR MI_02).
"""

from __future__ import annotations

import ctypes
import re
import sys
from ctypes import wintypes


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", wintypes.BYTE * 8),
    ]


class SP_DEVICE_INTERFACE_DATA(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("InterfaceClassGuid", GUID),
        ("Flags", wintypes.DWORD),
        ("Reserved", ctypes.POINTER(ctypes.c_ulong)),
    ]


class SP_DEVINFO_DATA(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("ClassGuid", GUID),
        ("DevInst", wintypes.DWORD),
        ("Reserved", ctypes.POINTER(ctypes.c_ulong)),
    ]


DIGCF_PRESENT = 0x00000002
DIGCF_DEVICEINTERFACE = 0x00000010
SPDRP_FRIENDLYNAME = 0x0000000C
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
_DETAIL_PATH_OFFSET = ctypes.sizeof(wintypes.DWORD)

# Same interface GUID embedded in MSMF symbolic links.
KSCATEGORY_VIDEO_CAMERA = "{E5323777-F976-4F5B-9B55-B94699C46E44}"


def _guid_from_str(s: str) -> GUID:
    ole32 = ctypes.WinDLL("ole32")
    g = GUID()
    hr = ole32.CLSIDFromString(ctypes.c_wchar_p(s), ctypes.byref(g))
    if hr:
        raise OSError(f"CLSIDFromString({s}) failed: 0x{hr & 0xFFFFFFFF:08X}")
    return g


def device_path_to_instance_id(path: str) -> str:
    r"""Convert a device-interface path to a PnP InstanceId-style string.

    \\?\usb#vid_0c45&pid_6366&mi_00#8&183af011&0&0000#{guid}\global
    -> USB\VID_0C45&PID_6366&MI_00\8&183AF011&0&0000
    """
    s = path.strip()
    if s.startswith("\\\\?\\"):
        s = s[4:]
    # Drop #{interface-guid}\rest
    m = re.search(r"#\{[0-9a-fA-F-]{36}\}", s)
    if m:
        s = s[: m.start()]
    return s.replace("#", "\\").upper()


def _list_video_interfaces_win():
    """Return [(msmf_index, device_id, friendly_name, device_path), ...]."""
    setupapi = ctypes.WinDLL("setupapi")

    SetupDiGetClassDevsW = setupapi.SetupDiGetClassDevsW
    SetupDiGetClassDevsW.argtypes = [
        ctypes.POINTER(GUID),
        wintypes.LPCWSTR,
        wintypes.HWND,
        wintypes.DWORD,
    ]
    SetupDiGetClassDevsW.restype = wintypes.HANDLE

    SetupDiEnumDeviceInterfaces = setupapi.SetupDiEnumDeviceInterfaces
    SetupDiEnumDeviceInterfaces.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.POINTER(GUID),
        wintypes.DWORD,
        ctypes.POINTER(SP_DEVICE_INTERFACE_DATA),
    ]
    SetupDiEnumDeviceInterfaces.restype = wintypes.BOOL

    SetupDiGetDeviceInterfaceDetailW = setupapi.SetupDiGetDeviceInterfaceDetailW
    SetupDiGetDeviceInterfaceDetailW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(SP_DEVICE_INTERFACE_DATA),
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(SP_DEVINFO_DATA),
    ]
    SetupDiGetDeviceInterfaceDetailW.restype = wintypes.BOOL

    SetupDiGetDeviceRegistryPropertyW = setupapi.SetupDiGetDeviceRegistryPropertyW
    SetupDiGetDeviceRegistryPropertyW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(SP_DEVINFO_DATA),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    SetupDiGetDeviceRegistryPropertyW.restype = wintypes.BOOL

    SetupDiDestroyDeviceInfoList = setupapi.SetupDiDestroyDeviceInfoList
    SetupDiDestroyDeviceInfoList.argtypes = [wintypes.HANDLE]
    SetupDiDestroyDeviceInfoList.restype = wintypes.BOOL

    iface = _guid_from_str(KSCATEGORY_VIDEO_CAMERA)
    hdev = SetupDiGetClassDevsW(
        ctypes.byref(iface), None, None, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE
    )
    if hdev in (0, None, INVALID_HANDLE_VALUE):
        return []

    detail_cb = 8 if ctypes.sizeof(ctypes.c_void_p) == 8 else 6
    devices = []
    try:
        index = 0
        while True:
            ifdata = SP_DEVICE_INTERFACE_DATA()
            ifdata.cbSize = ctypes.sizeof(SP_DEVICE_INTERFACE_DATA)
            if not SetupDiEnumDeviceInterfaces(
                hdev, None, ctypes.byref(iface), index, ctypes.byref(ifdata)
            ):
                break

            needed = wintypes.DWORD(0)
            SetupDiGetDeviceInterfaceDetailW(
                hdev, ctypes.byref(ifdata), None, 0, ctypes.byref(needed), None
            )
            if needed.value < _DETAIL_PATH_OFFSET + 2:
                index += 1
                continue

            buf = ctypes.create_string_buffer(needed.value)
            ctypes.memmove(buf, ctypes.byref(wintypes.DWORD(detail_cb)), 4)
            devinfo = SP_DEVINFO_DATA()
            devinfo.cbSize = ctypes.sizeof(SP_DEVINFO_DATA)
            if not SetupDiGetDeviceInterfaceDetailW(
                hdev,
                ctypes.byref(ifdata),
                buf,
                needed.value,
                None,
                ctypes.byref(devinfo),
            ):
                index += 1
                continue

            # DevicePath follows the DWORD cbSize directly (WCHAR needs only 2-byte
            # alignment). cbSize is 8 on x64 only because sizeof() pads the struct
            # tail; reading at offset 8 skips the leading "\\" of "\\?\".
            path = ctypes.wstring_at(ctypes.addressof(buf) + _DETAIL_PATH_OFFSET)
            if not path.startswith("\\\\?\\"):
                index += 1
                continue

            device_id = device_path_to_instance_id(path)
            name = f"Camera {index}"
            prop_type = wintypes.DWORD(0)
            name_needed = wintypes.DWORD(0)
            SetupDiGetDeviceRegistryPropertyW(
                hdev,
                ctypes.byref(devinfo),
                SPDRP_FRIENDLYNAME,
                ctypes.byref(prop_type),
                None,
                0,
                ctypes.byref(name_needed),
            )
            if name_needed.value:
                name_buf = ctypes.create_unicode_buffer(name_needed.value // 2 + 1)
                if SetupDiGetDeviceRegistryPropertyW(
                    hdev,
                    ctypes.byref(devinfo),
                    SPDRP_FRIENDLYNAME,
                    ctypes.byref(prop_type),
                    name_buf,
                    name_needed.value,
                    None,
                ):
                    if name_buf.value:
                        name = name_buf.value

            devices.append((index, device_id, name, path))
            index += 1
    finally:
        SetupDiDestroyDeviceInfoList(hdev)

    return devices


def list_capture_devices():
    """Stable capture endpoints for the active OpenCV backend on this platform.

    Returns list of dicts: {index, device_id, name}.
    On Windows, index is the SetupAPI/KSCATEGORY_VIDEO_CAMERA order used as the
    CAP_MSMF index. Returns None when enumeration is unavailable (non-Windows or
    SetupAPI failure) so callers can tell "cannot know" apart from "no devices".
    """
    if sys.platform != "win32":
        return None
    try:
        return [
            {"index": idx, "device_id": did, "name": name}
            for idx, did, name, _path in _list_video_interfaces_win()
        ]
    except OSError:
        return None


def find_index_for_device_id(devices, device_id: str | None):
    """Return the capture index of device_id within devices, or None if absent."""
    if not device_id or not devices:
        return None
    want = device_id.upper()
    for entry in devices:
        if entry["device_id"].upper() == want:
            return entry["index"]
    return None


if __name__ == "__main__":
    found = list_capture_devices()
    if found is None:
        print("Capture-device enumeration unavailable on this platform.")
    else:
        for entry in found:
            print(f"[{entry['index']}] {entry['name']}: {entry['device_id']}")
