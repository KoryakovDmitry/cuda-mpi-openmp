import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # Add known arguments
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--config", type=str)
    parser.add_argument("--timeout", type=int)

    # Parse known and unknown args
    args, unknown = parser.parse_known_args()

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

    return args, kwargs


if __name__ == "__main__":
    args, kwargs = parse_args()
    print("Known args:", args)
    print("Extra kwargs:", kwargs)
