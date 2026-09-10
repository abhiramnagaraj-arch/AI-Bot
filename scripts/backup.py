"""Consistent offline backup/restore. Refuses a running app or a nonempty restore target."""

import argparse
import fcntl
import os
import tarfile
from pathlib import Path


def backup(source: Path, destination: Path):
    source, destination = source.resolve(), destination.resolve()
    if not source.is_dir() or not (source / "metadata.sqlite3").is_file():
        raise ValueError("Source is not an application runtime directory")
    if destination.is_relative_to(source):
        raise ValueError("Write the backup outside the runtime directory")
    with (source / "application.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Stop the application before backing up its runtime") from None
        fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with (
                os.fdopen(fd, "wb") as stream,
                tarfile.open(fileobj=stream, mode="w:gz") as archive,
            ):
                for path in sorted(source.rglob("*")):
                    if path.is_symlink():
                        raise ValueError("Runtime backups do not follow symbolic links")
                    if path.is_file() and path.name != "application.lock":
                        archive.add(path, arcname=str(path.relative_to(source)), recursive=False)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise


def restore(archive_path: Path, destination: Path):
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Restore requires a new or empty directory")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if not any(m.name == "metadata.sqlite3" for m in members):
            raise ValueError("Archive does not contain application metadata")
        for member in members:
            path = destination / member.name
            if not member.isfile() or not path.resolve().is_relative_to(destination):
                raise ValueError("Archive contains an unsafe path or file type")
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Trusted operator-provided backups only; extraction does not overwrite user data.
        archive.extractall(destination, members=members, filter="data")
    # Rebase absolute source paths after a restore into a different directory.
    import sqlite3

    with sqlite3.connect(destination / "metadata.sqlite3") as db:
        for vid, old in db.execute("SELECT id,path FROM versions").fetchall():
            db.execute(
                "UPDATE versions SET path=? WHERE id=?",
                (str(destination / "uploads" / Path(old).name), vid),
            )
        db.execute("DELETE FROM auth_tokens")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["backup", "restore"])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    (backup if args.operation == "backup" else restore)(args.source, args.destination)
    print(f"{args.operation.capitalize()} completed: {args.destination}")


if __name__ == "__main__":
    main()
