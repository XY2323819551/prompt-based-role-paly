import platform
import sys


def get_os_version():
    """
    Determines the operating system version of the current system.

    This function checks the operating system of the current environments and attempts
    to return a human-readable version string. For macOS, it uses the `platform.mac_ver()`
    method. For Linux, it attempts to read the version information from `/etc/os-release`.
    If the system is not macOS or Linux, or if the Linux version cannot be determined, it
    defaults to a generic version string or "Unknown Operating System".

    Returns:
        str: A string describing the operating system version, or "Unknown Operating System"
             if the version cannot be determined.
    """
    system = platform.system()

    if system == "Darwin":
        # macOS
        return 'macOS ' + platform.mac_ver()[0]
    elif system == "Linux":
        try:
            with open("/etc/os-release") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("PRETTY_NAME"):
                        return line.split("=")[1].strip().strip('"')
        except FileNotFoundError:
            pass

        return platform.version()
    elif system == "Windows":
        return f"Windows {platform.release()}"
    else:
        return "Unknown Operating System"


def get_python_version():
    """
    Returns the current Python version.

    Returns:
        str: Current Python version
    """
    return f"Python {sys.version.split()[0]}"


def get_system_info():
    """
    Returns a formatted string containing both OS and Python version information.

    Returns:
        str: Formatted string with system information
    """
    os_ver = get_os_version()
    py_ver = get_python_version()
    return f"Operating System: {os_ver}\nPython Version: {py_ver}"


if __name__ == "__main__":
    print(get_system_info())

