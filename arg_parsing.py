def hundle_unkown(unknown):
    # Process unknown args into kwargs
    kwargs = {}
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith("--"):
            key = arg.lstrip("--")
            value = True  # Default value for flags
            # Check if next arg exists and is not another flag
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                i += 1
                value = unknown[i]
                # Attempt to convert value to bool, int, or float
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # Keep as string
            kwargs[key] = value
        else:
            # Handle positional arguments or ignore
            pass
        i += 1

    return kwargs
