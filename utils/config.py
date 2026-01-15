import configparser


def config(section: str, key: str, conf_file: str = "project.conf") -> str:
    """
    Get a value from a .conf file (INI-style).

    Args:
        section (str): Section name in the .conf file
        key (str): Key name in the section
        conf_file (str): Path to the .conf file relative to project root

    Returns:
        str: Value as string. Raises KeyError if section/key not found.
    """
    parser = configparser.ConfigParser()
    parser.read(conf_file)

    if not parser.has_section(section):
        raise KeyError(f"Section '{section}' not found in {conf_file}")
    if not parser.has_option(section, key):
        raise KeyError(f"Key '{key}' not found in section '{section}' of {conf_file}")

    return parser.get(section, key)
