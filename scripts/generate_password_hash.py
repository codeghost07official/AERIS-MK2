from getpass import getpass

from werkzeug.security import generate_password_hash


def main() -> None:
    password = getpass("Password to hash: ")
    confirmation = getpass("Confirm password: ")
    if not password:
        raise SystemExit("Password cannot be empty.")
    if password != confirmation:
        raise SystemExit("Passwords did not match.")
    print(generate_password_hash(password))


if __name__ == "__main__":
    main()
