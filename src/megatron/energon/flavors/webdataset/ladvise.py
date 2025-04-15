import ctypes
import os

# Load Lustre API library (adjust path if needed)
liblustreapi = ctypes.CDLL("liblustreapi.so", mode=ctypes.RTLD_GLOBAL)

# Define constants (adjust based on your Lustre version)
LU_ADVISE_WILLREAD = 1  # Example value for LU_LADVISE_WILLREAD
LU_ADVISE_RANDOM = 2  # Example value for LU_LADVISE_RANDOM
LF_ASYNC = 0x01  # Flag for asynchronous advice processing


# Define the llapi_lu_ladvise structure
class llapi_lu_ladvise(ctypes.Structure):
    _fields_ = [
        ("lla_advice", ctypes.c_ushort),
        ("lla_value1", ctypes.c_ushort),
        ("lla_value2", ctypes.c_uint),
        ("lla_start", ctypes.c_ulonglong),
        ("lla_end", ctypes.c_ulonglong),
        ("lla_value3", ctypes.c_uint),
        ("lla_value4", ctypes.c_uint),
    ]


# Define llapi_ladvise function signature
liblustreapi.llapi_ladvise.argtypes = [
    ctypes.c_int,  # fd
    ctypes.c_ulonglong,  # flags
    ctypes.c_int,  # num_advise
    ctypes.POINTER(llapi_lu_ladvise),  # ladvise array
]
liblustreapi.llapi_ladvise.restype = ctypes.c_int


def set_lustre_advice(file_path, advice_type, start_offset, end_offset):
    # Open the Lustre file
    fd = os.open(file_path, os.O_RDWR)
    if fd < 0:
        raise OSError(f"Failed to open {file_path}")

    # Configure advice parameters
    ladvise = llapi_lu_ladvise()
    ladvise.lla_advice = advice_type
    ladvise.lla_start = start_offset
    ladvise.lla_end = end_offset

    # Call llapi_ladvise (1 advice struct, no flags)
    rc = liblustreapi.llapi_ladvise(
        fd,
        0,  # flags (0 for synchronous)
        1,  # num_advise entries
        ctypes.byref(ladvise),
    )

    os.close(fd)
    return rc


# Example usage
if __name__ == "__main__":
    rc = set_lustre_advice("/mnt/lustre/myfile", LU_ADVISE_WILLREAD, 0, 1048576)
    print(f"llapi_ladvise returned: {rc}")
