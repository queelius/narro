"""Tests for venv creation + pip install helpers."""
import errno
import fcntl
import io
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import muse.core.venv as venv_module
from muse.core.venv import (
    create_venv,
    install_python_sources,
    install_output_mode,
    install_into_venv,
    run_owned_command,
    venv_transaction,
    venv_python,
    find_free_port,
)


_TRELLIS_REVISION = "442aa1e1afb9014e80681d3bf604e8d728a86ee7"
_FLEXICUBES_REVISION = "815e075a2a400d06c48d94c347674344ed6ae5c5"


def _trellis_source_spec():
    return {
        "type": "git",
        "name": "trellis",
        "url": "https://github.com/microsoft/TRELLIS.git",
        "revision": _TRELLIS_REVISION,
        "sparse_paths": ("trellis",),
        "required_paths": (
            "trellis/__init__.py",
            "trellis/pipelines/trellis_image_to_3d.py",
            "trellis/representations/mesh/flexicubes/flexicubes.py",
        ),
        "pth_path": ".",
        "submodules": ({
            "path": "trellis/representations/mesh/flexicubes",
            "url": "https://github.com/MaxtirError/FlexiCubes.git",
            "revision": _FLEXICUBES_REVISION,
        },),
    }


def _empty_test_venv(tmp_path):
    venv_path = tmp_path / "venv"
    venv_module._venv_site_packages(venv_path).mkdir(parents=True)
    return venv_path


def _complete_test_venv(path, marker="prior"):
    python = venv_python(path)
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o700)
    (path / "marker").write_text(marker, encoding="utf-8")
    return path


def _mock_source_git(commands):
    """Return a filesystem-only fake for the owned Git command runner."""
    spec = _trellis_source_spec()

    def run(command, **kwargs):
        command = list(command)
        checkout = Path(command[command.index("-C") + 1])
        git_args = tuple(command[command.index("-C") + 2:])
        commands.append((checkout, git_args, kwargs))

        if git_args[:1] == ("checkout",):
            (checkout / "trellis" / "pipelines").mkdir(parents=True)
            (checkout / "trellis" / "__init__.py").write_text("", encoding="utf-8")
            (checkout / "trellis" / "pipelines" / "trellis_image_to_3d.py").write_text(
                "", encoding="utf-8",
            )
            (checkout / ".gitmodules").write_text(
                '[submodule "trellis/representations/mesh/flexicubes"]\n'
                "\tpath = trellis/representations/mesh/flexicubes\n"
                "\turl = https://github.com/MaxtirError/FlexiCubes.git\n",
                encoding="utf-8",
            )
        elif git_args[:2] == ("submodule", "update"):
            flexicubes = (
                checkout / "trellis" / "representations" / "mesh" / "flexicubes"
            )
            flexicubes.mkdir(parents=True)
            (flexicubes / "flexicubes.py").write_text("", encoding="utf-8")

        stdout = ""
        if git_args == ("remote", "get-url", "origin"):
            stdout = spec["url"]
        elif git_args == ("rev-parse", "HEAD"):
            stdout = (
                _FLEXICUBES_REVISION
                if checkout.name == "flexicubes"
                else _TRELLIS_REVISION
            )
        elif git_args == ("status", "--porcelain=v1", "--untracked-files=all"):
            if (checkout / "untracked.py").exists():
                stdout = "?? untracked.py"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    return run


def _mock_process(
    *, returncode=0, stdout: bytes = b"", stderr: bytes = b"", pid=4242,
):
    process = MagicMock()
    process.pid = pid
    process.stdout = io.BytesIO(stdout)
    process.stderr = io.BytesIO(stderr)
    process.wait.return_value = returncode
    process.poll.return_value = returncode
    return process


def _group_signal_that_exits_on(
    terminal_signal: signal.Signals,
) -> tuple[list[tuple[int, signal.Signals | int]], object, dict[str, bool]]:
    alive = {"value": True}
    calls: list[tuple[int, signal.Signals | int]] = []

    def killpg(process_group, sig):
        calls.append((process_group, sig))
        if sig == 0 and not alive["value"]:
            raise ProcessLookupError
        if sig == terminal_signal:
            alive["value"] = False

    return calls, killpg, alive


class TestPythonSources:
    def test_materializes_exact_sparse_commits_and_writes_venv_pth(
        self, tmp_path, monkeypatch,
    ):
        venv_path = _empty_test_venv(tmp_path)
        commands = []
        monkeypatch.setenv("GIT_DIR", "/attacker-controlled")
        monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'url.evil.insteadOf=https://'")

        with patch(
            "muse.core.venv._run_owned", side_effect=_mock_source_git(commands),
        ):
            installed = install_python_sources(venv_path, [_trellis_source_spec()])

        checkout = venv_path / "muse-sources" / "trellis" / _TRELLIS_REVISION
        assert installed == (checkout,)
        assert (
            venv_module._venv_site_packages(venv_path)
            / "muse-source-trellis.pth"
        ).read_text(encoding="utf-8") == f"{checkout}\n"

        git_args = [call[1] for call in commands]
        assert (
            "fetch", "--depth", "1", "--filter=blob:none", "--no-tags",
            "origin", _TRELLIS_REVISION,
        ) in git_args
        assert (
            "submodule", "update", "--init", "--depth", "1",
            "--filter=blob:none", "--",
            "trellis/representations/mesh/flexicubes",
        ) in git_args
        assert all(call[2]["capture_output"] is True for call in commands)
        assert all(call[2]["timeout"] <= venv_module._SOURCE_INSTALL_TIMEOUT for call in commands)
        assert all("GIT_DIR" not in call[2]["env"] for call in commands)
        assert all("GIT_CONFIG_PARAMETERS" not in call[2]["env"] for call in commands)
        assert all(call[2]["env"]["GIT_CONFIG_GLOBAL"] == os.devnull for call in commands)
        assert all(call[2]["env"]["GIT_CONFIG_NOSYSTEM"] == "1" for call in commands)
        assert all(call[2]["env"]["GIT_TERMINAL_PROMPT"] == "0" for call in commands)
        assert "core.fsmonitor=false" in venv_module._GIT_SAFE_CONFIG
        assert "core.untrackedCache=false" in venv_module._GIT_SAFE_CONFIG
        assert any(
            args == ("status", "--porcelain=v1", "--untracked-files=all")
            and checkout.name == "flexicubes"
            for checkout, args, _ in commands
        )

    def test_empty_declarations_revoke_removed_source_hooks(self, tmp_path):
        venv_path = _empty_test_venv(tmp_path)
        site_packages = venv_module._venv_site_packages(venv_path)
        stale = site_packages / "muse-source-removed.pth"
        stale.write_text("/reviewed/source/that-was-removed\n", encoding="utf-8")

        assert install_python_sources(venv_path, []) == ()

        assert not stale.exists()

    def test_replacement_revokes_renamed_source_hook_before_fetch(self, tmp_path):
        venv_path = _empty_test_venv(tmp_path)
        site_packages = venv_module._venv_site_packages(venv_path)
        stale = site_packages / "muse-source-old-name.pth"
        stale.write_text("/old/reviewed/source\n", encoding="utf-8")
        invalid = _trellis_source_spec()
        invalid["revision"] = "main"

        with pytest.raises(ValueError), patch("muse.core.venv._run_owned") as run:
            install_python_sources(venv_path, [invalid])

        assert not stale.exists()
        run.assert_not_called()

    def test_stale_source_symlink_is_unlinked_without_touching_target(self, tmp_path):
        venv_path = _empty_test_venv(tmp_path)
        site_packages = venv_module._venv_site_packages(venv_path)
        outside = tmp_path / "outside.pth"
        outside.write_text("keep\n", encoding="utf-8")
        stale = site_packages / "muse-source-stale.pth"
        stale.symlink_to(outside)

        install_python_sources(venv_path, [])

        assert not stale.exists()
        assert outside.read_text(encoding="utf-8") == "keep\n"

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda spec: spec.update(revision="main"),
            lambda spec: spec.update(url="ssh://github.com/microsoft/TRELLIS.git"),
            lambda spec: spec.update(sparse_paths=("../trellis",)),
            lambda spec: spec["submodules"][0].update(path="outside"),
        ],
    )
    def test_rejects_unsafe_source_declarations_before_git(self, tmp_path, mutate):
        venv_path = _empty_test_venv(tmp_path)
        valid = _trellis_source_spec()
        invalid = _trellis_source_spec()
        mutate(invalid)

        with patch("muse.core.venv._run_owned") as run:
            with pytest.raises(ValueError):
                install_python_sources(venv_path, [valid, invalid])

        run.assert_not_called()
        assert not (venv_path / "muse-sources").exists()

    def test_failed_fetch_cleans_only_owned_staging_directory(self, tmp_path):
        venv_path = _empty_test_venv(tmp_path)
        marker = venv_path / "keep-me"
        marker.write_text("user state", encoding="utf-8")
        commands = []
        fake_git = _mock_source_git(commands)

        def fail_fetch(command, **kwargs):
            git_args = tuple(command[list(command).index("-C") + 2:])
            if git_args[:1] == ("fetch",):
                raise subprocess.CalledProcessError(
                    1, list(command), stderr="mocked fetch failure",
                )
            return fake_git(command, **kwargs)

        with patch("muse.core.venv._run_owned", side_effect=fail_fetch):
            with pytest.raises(subprocess.CalledProcessError):
                install_python_sources(venv_path, [_trellis_source_spec()])

        assert marker.read_text(encoding="utf-8") == "user state"
        name_root = venv_path / "muse-sources" / "trellis"
        assert name_root.is_dir()
        assert list(name_root.iterdir()) == []
        assert not (
            venv_module._venv_site_packages(venv_path)
            / "muse-source-trellis.pth"
        ).exists()

    def test_dirty_existing_checkout_is_rejected_and_disabled(self, tmp_path):
        venv_path = _empty_test_venv(tmp_path)
        commands = []
        fake_git = _mock_source_git(commands)
        spec = _trellis_source_spec()

        with patch("muse.core.venv._run_owned", side_effect=fake_git):
            checkout, = install_python_sources(venv_path, [spec])

        pth = (
            venv_module._venv_site_packages(venv_path)
            / "muse-source-trellis.pth"
        )
        assert pth.is_file()
        (checkout / "untracked.py").write_text("malicious = True", encoding="utf-8")

        with patch("muse.core.venv._run_owned", side_effect=fake_git):
            with pytest.raises(RuntimeError, match="local changes"):
                install_python_sources(venv_path, [spec])

        assert not pth.exists(), "a rejected checkout must no longer be importable"

    def test_partial_multi_source_activation_is_rolled_back(self, tmp_path):
        venv_path = _empty_test_venv(tmp_path)
        commands = []
        first = _trellis_source_spec()
        second = _trellis_source_spec()
        second["name"] = "trellis-second"
        real_write = venv_module._write_source_pth

        def fail_second_write(path, checkout, spec):
            if spec.name == "trellis-second":
                raise OSError("mocked activation failure")
            return real_write(path, checkout, spec)

        with patch(
            "muse.core.venv._run_owned", side_effect=_mock_source_git(commands),
        ), patch(
            "muse.core.venv._write_source_pth", side_effect=fail_second_write,
        ):
            with pytest.raises(OSError, match="activation failure"):
                install_python_sources(venv_path, [first, second])

        site_packages = venv_module._venv_site_packages(venv_path)
        assert not (site_packages / "muse-source-trellis.pth").exists()
        assert not (site_packages / "muse-source-trellis-second.pth").exists()

    def test_symlink_substitution_is_unlinked_without_traversing_target(self, tmp_path):
        venv_path = _empty_test_venv(tmp_path)
        external = tmp_path / "external"
        external.mkdir()
        marker = external / "keep"
        marker.write_text("valuable", encoding="utf-8")
        commands = []
        fake_git = _mock_source_git(commands)

        def substitute_at_fetch(command, **kwargs):
            command = list(command)
            checkout = Path(command[command.index("-C") + 1])
            git_args = tuple(command[command.index("-C") + 2:])
            if git_args[:1] == ("fetch",):
                checkout.rmdir()
                checkout.symlink_to(external, target_is_directory=True)
                raise subprocess.CalledProcessError(1, command)
            return fake_git(command, **kwargs)

        with patch("muse.core.venv._run_owned", side_effect=substitute_at_fetch), \
             patch("muse.core.venv.shutil.rmtree") as rmtree:
            with pytest.raises(subprocess.CalledProcessError):
                install_python_sources(venv_path, [_trellis_source_spec()])

        assert marker.read_text(encoding="utf-8") == "valuable"
        assert list((venv_path / "muse-sources" / "trellis").iterdir()) == []
        rmtree.assert_not_called()


class TestVenvPython:
    def test_returns_bin_python_on_posix(self, tmp_path):
        # On POSIX venv layout, python is at <venv>/bin/python
        path = venv_python(tmp_path)
        assert path == tmp_path / "bin" / "python"


class TestCreateVenv:
    def test_managed_job_child_does_not_escape_outer_process_group(self):
        process = _mock_process()
        with patch("muse.core.venv._OS_NAME", "posix"), \
             patch.dict(
                 os.environ,
                 {"MUSE_MANAGED_JOB_PROCESS_GROUP": "1"},
             ), \
             patch("muse.core.venv.os.getpgid", return_value=31337), \
             patch("muse.core.venv.os.getpgrp", return_value=31337), \
             patch("muse.core.venv.subprocess.Popen", return_value=process) as popen:
            owned = venv_module._spawn_owned(["mock-installer"], capture_output=False)

        assert owned.process is process
        assert "start_new_session" not in popen.call_args.kwargs

    def test_calls_python_venv_module_in_isolated_group(self, tmp_path):
        process = _mock_process()
        target = tmp_path / "myenv"
        with patch("muse.core.venv._OS_NAME", "posix"), \
             patch("muse.core.venv.os.getpgid", return_value=4242), \
             patch("muse.core.venv.os.getpgrp", return_value=31337), \
             patch("muse.core.venv.os.killpg", side_effect=ProcessLookupError), \
             patch("muse.core.venv._validate_created_venv"), \
             patch("muse.core.venv.subprocess.Popen", return_value=process) as popen:
            create_venv(target)

        popen.assert_called_once()
        args = popen.call_args.args[0]
        # Use sys.executable to guarantee we create the venv with the same
        # Python that muse is running on (matters for ABI compatibility)
        import sys
        assert args[0] == sys.executable
        assert "-m" in args and "venv" in args
        staging = Path(args[-1])
        assert staging.parent == target.parent
        assert staging.name.startswith(".myenv.staging-")
        assert str(target) not in args
        assert target.is_dir()
        assert not staging.exists()
        assert popen.call_args.kwargs["start_new_session"] is True
        assert popen.call_args.kwargs["stdout"] is subprocess.PIPE
        assert popen.call_args.kwargs["stderr"] is subprocess.PIPE
        assert popen.call_args.kwargs["bufsize"] == 0
        process.wait.assert_called_once_with(timeout=venv_module._VENV_CREATE_TIMEOUT)

    def test_completed_failure_never_signals_reusable_group(self, tmp_path, capsys):
        process = _mock_process(returncode=1, stdout=b"venv out\n", stderr=b"venv err\n")
        signals, killpg, _ = _group_signal_that_exits_on(signal.SIGTERM)
        with patch("muse.core.venv._OS_NAME", "posix"), \
             patch("muse.core.venv.os.getpgid", return_value=4242), \
             patch("muse.core.venv.os.getpgrp", return_value=31337), \
             patch("muse.core.venv.os.killpg", side_effect=killpg), \
             patch("muse.core.venv.subprocess.Popen", return_value=process):
            with pytest.raises(subprocess.CalledProcessError) as exc_info:
                create_venv(tmp_path / "doomed")

        assert signals == []
        assert exc_info.value.stdout == "venv out\n"
        assert exc_info.value.stderr == "venv err\n"
        assert capsys.readouterr().err == "venv out\nvenv err\n"
        assert not (tmp_path / "doomed").exists()
        assert list(tmp_path.glob(".doomed.staging-*")) == []

    def test_timeout_cleans_descendant_group_and_preserves_bounded_output(self, tmp_path):
        process = _mock_process(stdout=b"partial out", stderr=b"partial err")
        process.wait.side_effect = [
            subprocess.TimeoutExpired(
                cmd=["python", "-m", "venv"], timeout=venv_module._VENV_CREATE_TIMEOUT,
            ),
            0,
        ]
        signals, killpg, alive = _group_signal_that_exits_on(signal.SIGTERM)
        process.poll.side_effect = lambda: None if alive["value"] else 0
        with patch("muse.core.venv._OS_NAME", "posix"), \
             patch("muse.core.venv.os.getpgid", return_value=4242), \
             patch("muse.core.venv.os.getpgrp", return_value=31337), \
             patch("muse.core.venv.os.killpg", side_effect=killpg), \
             patch("muse.core.venv.subprocess.Popen", return_value=process):
            with pytest.raises(subprocess.TimeoutExpired) as exc_info:
                create_venv(tmp_path / "slow")

        assert (4242, signal.SIGTERM) in signals
        assert exc_info.value.stdout == "partial out"
        assert exc_info.value.stderr == "partial err"
        assert not (tmp_path / "slow").exists()
        assert list(tmp_path.glob(".slow.staging-*")) == []

    def test_keyboard_interrupt_cleans_descendant_group(self, tmp_path):
        process = _mock_process()
        process.wait.side_effect = [KeyboardInterrupt, 0]
        signals, killpg, alive = _group_signal_that_exits_on(signal.SIGTERM)
        process.poll.side_effect = lambda: None if alive["value"] else 0
        with patch("muse.core.venv._OS_NAME", "posix"), \
             patch("muse.core.venv.os.getpgid", return_value=4242), \
             patch("muse.core.venv.os.getpgrp", return_value=31337), \
             patch("muse.core.venv.os.killpg", side_effect=killpg), \
             patch("muse.core.venv.subprocess.Popen", return_value=process):
            with pytest.raises(KeyboardInterrupt):
                create_venv(tmp_path / "interrupted")

        assert (4242, signal.SIGTERM) in signals
        assert not (tmp_path / "interrupted").exists()
        assert list(tmp_path.glob(".interrupted.staging-*")) == []

    def test_success_validates_and_atomically_promotes_staging(self, tmp_path):
        target = tmp_path / "ready"

        def fake_run(command, **_kwargs):
            staging = Path(command[-1])
            python = venv_python(staging)
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\n")
            python.chmod(0o700)
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch("muse.core.venv.run_owned_command", side_effect=fake_run):
            create_venv(target)

        assert venv_python(target).is_file()
        assert list(tmp_path.glob(".ready.staging-*")) == []

    def test_refuses_existing_or_dangling_final_path(self, tmp_path):
        target = tmp_path / "existing"
        target.symlink_to(tmp_path / "missing")

        with pytest.raises(FileExistsError, match="replace existing"):
            create_venv(target)

        assert target.is_symlink()

    def test_success_without_interpreter_removes_staging(self, tmp_path):
        target = tmp_path / "incomplete"
        completed = subprocess.CompletedProcess([], 0, "", "")

        with patch("muse.core.venv.run_owned_command", return_value=completed):
            with pytest.raises(RuntimeError, match="without an executable"):
                create_venv(target)

        assert not target.exists()
        assert list(tmp_path.glob(".incomplete.staging-*")) == []


class TestEnsureVenv:
    def test_missing_path_uses_creator(self, tmp_path):
        target = tmp_path / "new"
        creator = MagicMock()

        venv_module.ensure_venv(target, creator=creator)

        creator.assert_called_once_with(target)

    def test_incomplete_existing_directory_fails_closed(self, tmp_path):
        target = tmp_path / "broken"
        target.mkdir()
        creator = MagicMock()

        with pytest.raises(RuntimeError, match="existing venv is incomplete"):
            venv_module.ensure_venv(target, creator=creator)

        creator.assert_not_called()
        assert target.is_dir()


class TestVenvTransaction:
    def test_uncommitted_update_restores_prior_venv(self, tmp_path):
        target = _complete_test_venv(tmp_path / "model")

        with venv_transaction(target) as transaction:
            assert transaction.path == target
            (target / "marker").write_text("mutated", encoding="utf-8")
            (target / "new-package").write_text("partial", encoding="utf-8")

        assert (target / "marker").read_text(encoding="utf-8") == "prior"
        assert not (target / "new-package").exists()
        assert list(tmp_path.glob(".model.transaction-*")) == []

    def test_commit_promotes_clone_at_canonical_path(self, tmp_path):
        target = _complete_test_venv(tmp_path / "model")

        with venv_transaction(target) as transaction:
            (target / "marker").write_text("updated", encoding="utf-8")
            transaction.commit()

        assert (target / "marker").read_text(encoding="utf-8") == "updated"
        assert list(tmp_path.glob(".model.transaction-*")) == []

    def test_exception_even_after_commit_restores_prior_venv(self, tmp_path):
        target = _complete_test_venv(tmp_path / "model")

        with pytest.raises(KeyboardInterrupt):
            with venv_transaction(target) as transaction:
                (target / "marker").write_text("mutated", encoding="utf-8")
                transaction.commit()
                raise KeyboardInterrupt

        assert (target / "marker").read_text(encoding="utf-8") == "prior"
        assert list(tmp_path.glob(".model.transaction-*")) == []

    def test_new_uncommitted_venv_is_removed(self, tmp_path):
        target = tmp_path / "model"

        with venv_transaction(target):
            _complete_test_venv(target, marker="partial")

        assert not target.exists()
        assert list(tmp_path.glob(".model.transaction-*")) == []

    def test_preflight_disk_failure_does_not_move_prior_venv(self, tmp_path):
        target = _complete_test_venv(tmp_path / "model")
        usage = MagicMock(free=10)

        with patch(
            "muse.core.venv._venv_clone_bytes", return_value=100,
        ), patch(
            "muse.core.venv.shutil.disk_usage", return_value=usage,
        ):
            with pytest.raises(RuntimeError, match="insufficient disk space"):
                with venv_transaction(target):
                    pass

        assert (target / "marker").read_text(encoding="utf-8") == "prior"
        assert list(tmp_path.glob(".model.transaction-*")) == []

    def test_copy_disk_failure_restores_prior_and_translates_error(self, tmp_path):
        target = _complete_test_venv(tmp_path / "model")

        def fail_copy(_source, destination, **_kwargs):
            Path(destination).mkdir()
            (Path(destination) / "partial").write_text("partial", encoding="utf-8")
            raise OSError(errno.ENOSPC, "mock disk full")

        with patch("muse.core.venv.shutil.copytree", side_effect=fail_copy):
            with pytest.raises(RuntimeError, match="insufficient disk space"):
                with venv_transaction(target):
                    pass

        assert (target / "marker").read_text(encoding="utf-8") == "prior"
        assert not (target / "partial").exists()
        assert list(tmp_path.glob(".model.transaction-*")) == []


class TestInstallIntoVenv:
    def test_uses_venvs_pip_not_system_pip(self, tmp_path):
        process = _mock_process()
        # Simulate a venv layout
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "python").touch()
        with patch("muse.core.venv._OS_NAME", "posix"), \
             patch("muse.core.venv.os.getpgid", return_value=4242), \
             patch("muse.core.venv.os.getpgrp", return_value=31337), \
             patch("muse.core.venv.os.killpg", side_effect=ProcessLookupError), \
             patch("muse.core.venv.subprocess.Popen", return_value=process) as popen:
            install_into_venv(tmp_path, ["numpy", "scipy"])
        args = popen.call_args.args[0]
        # Must be <venv>/bin/python -m pip install <pkgs>
        assert args[0] == str(tmp_path / "bin" / "python")
        assert args[1:4] == ["-m", "pip", "install"]
        assert args[4] == "-q"
        assert "numpy" in args
        assert "scipy" in args
        assert popen.call_args.kwargs["start_new_session"] is True
        assert popen.call_args.kwargs["stdout"] is subprocess.PIPE
        assert popen.call_args.kwargs["stderr"] is subprocess.PIPE
        assert popen.call_args.kwargs["bufsize"] == 0

    @patch("muse.core.venv.subprocess.Popen")
    def test_empty_package_list_is_noop(self, mock_popen, tmp_path):
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "python").touch()
        install_into_venv(tmp_path, [])
        mock_popen.assert_not_called()

    def test_completed_install_failure_does_not_signal_group(self, tmp_path):
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "python").touch()
        process = _mock_process(returncode=1)
        signals, killpg, _ = _group_signal_that_exits_on(signal.SIGTERM)
        with patch("muse.core.venv._OS_NAME", "posix"), \
             patch("muse.core.venv.os.getpgid", return_value=4242), \
             patch("muse.core.venv.os.getpgrp", return_value=31337), \
             patch("muse.core.venv.os.killpg", side_effect=killpg), \
             patch("muse.core.venv.subprocess.Popen", return_value=process):
            with pytest.raises(subprocess.CalledProcessError):
                install_into_venv(tmp_path, ["bogus"])
        assert signals == []

    def test_verbose_mode_streams_output_and_keeps_group_ownership(self, tmp_path):
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "python").touch()
        process = _mock_process()
        with patch("muse.core.venv._OS_NAME", "posix"), \
             patch("muse.core.venv.os.getpgid", return_value=4242), \
             patch("muse.core.venv.os.getpgrp", return_value=31337), \
             patch("muse.core.venv.os.killpg", side_effect=ProcessLookupError), \
             patch("muse.core.venv.subprocess.Popen", return_value=process) as popen, \
             install_output_mode(verbose=True):
            install_into_venv(tmp_path, ["numpy"])

        args = popen.call_args.args[0]
        assert "-q" not in args
        assert popen.call_args.kwargs == {"start_new_session": True}
        process.wait.assert_called_once_with(timeout=venv_module._PIP_TIMEOUT)

    def test_success_never_signals_reusable_group_identifier(self, tmp_path):
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "python").touch()
        process = _mock_process(returncode=0)
        signals, killpg, _ = _group_signal_that_exits_on(signal.SIGTERM)
        with patch("muse.core.venv._OS_NAME", "posix"), \
             patch("muse.core.venv.os.getpgid", return_value=4242), \
             patch("muse.core.venv.os.getpgrp", return_value=31337), \
             patch("muse.core.venv.os.killpg", side_effect=killpg), \
             patch("muse.core.venv.subprocess.Popen", return_value=process):
            install_into_venv(tmp_path, ["numpy"])

        assert signals == []

    def test_group_that_ignores_term_is_killed_within_shared_budget(self, tmp_path):
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "python").touch()
        process = _mock_process(returncode=1)
        process.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="pip", timeout=venv_module._PIP_TIMEOUT),
            1,
        ]
        signals, killpg, alive = _group_signal_that_exits_on(signal.SIGKILL)
        process.poll.side_effect = lambda: None if alive["value"] else 1
        with patch("muse.core.venv._OS_NAME", "posix"), \
             patch("muse.core.venv.os.getpgid", return_value=4242), \
             patch("muse.core.venv.os.getpgrp", return_value=31337), \
             patch("muse.core.venv.os.killpg", side_effect=killpg), \
             patch("muse.core.venv.subprocess.Popen", return_value=process), \
             patch("muse.core.venv._CLEANUP_TIMEOUT", 0.0):
            with pytest.raises(subprocess.TimeoutExpired):
                install_into_venv(tmp_path, ["bogus"])

        assert (4242, signal.SIGTERM) in signals
        assert (4242, signal.SIGKILL) in signals

    def test_invalid_mock_pid_never_reaches_signal_methods(self, tmp_path):
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "python").touch()
        process = _mock_process(returncode=1, pid=MagicMock(name="unsafe-pid"))
        process.poll.return_value = None
        with patch("muse.core.venv._OS_NAME", "posix"), \
             patch("muse.core.venv.os.killpg") as killpg, \
             patch("muse.core.venv.subprocess.Popen", return_value=process), \
             patch("muse.core.venv._CLEANUP_TIMEOUT", 0.0):
            with pytest.raises(subprocess.CalledProcessError):
                install_into_venv(tmp_path, ["bogus"])

        killpg.assert_not_called()
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_unverified_group_falls_back_to_exact_popen_handle(self, tmp_path):
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "python").touch()
        process = _mock_process(returncode=1)
        alive = {"value": True}
        process.poll.side_effect = lambda: None if alive["value"] else 1
        process.terminate.side_effect = lambda: alive.update(value=False)
        process.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="pip", timeout=venv_module._PIP_TIMEOUT),
            1,
        ]
        with patch("muse.core.venv.os.name", "posix"), \
             patch("muse.core.venv.os.getpgid", return_value=31337), \
             patch("muse.core.venv.os.getpgrp", return_value=31337), \
             patch("muse.core.venv.os.killpg") as killpg, \
             patch("muse.core.venv.subprocess.Popen", return_value=process):
            with pytest.raises(subprocess.TimeoutExpired):
                install_into_venv(tmp_path, ["bogus"])

        process.terminate.assert_called_once_with()
        process.kill.assert_not_called()
        killpg.assert_not_called()

    def test_group_revalidation_exit_never_falls_back_to_reusable_pid(self):
        process = _mock_process(returncode=1)
        process.poll.side_effect = [None, 1]
        target = venv_module._OwnedProcess(
            process=process,
            pid=4242,
            process_group=4242,
        )

        with patch("muse.core.venv._OS_NAME", "posix"), \
             patch("muse.core.venv.os.getpgrp", return_value=31337), \
             patch("muse.core.venv.os.killpg") as killpg:
            signalled = venv_module._signal_owned(
                target, signal.SIGTERM, force=False,
            )

        assert signalled is False
        assert process.poll.call_count == 2
        killpg.assert_not_called()
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_windows_uses_new_group_flag_and_exact_handle_fallback(self, tmp_path):
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "python").touch()
        process = _mock_process(returncode=1)
        alive = {"value": True}
        process.poll.side_effect = lambda: None if alive["value"] else 1
        process.terminate.side_effect = lambda: alive.update(value=False)
        process.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="pip", timeout=venv_module._PIP_TIMEOUT),
            1,
        ]
        create_group = 0x200
        with patch("muse.core.venv._OS_NAME", "nt"), \
             patch(
                 "muse.core.venv.subprocess.CREATE_NEW_PROCESS_GROUP",
                 create_group,
                 create=True,
             ), \
             patch("muse.core.venv.os.killpg") as killpg, \
             patch("muse.core.venv.subprocess.Popen", return_value=process) as popen:
            with pytest.raises(subprocess.TimeoutExpired):
                install_into_venv(tmp_path, ["bogus"])

        assert popen.call_args.kwargs["creationflags"] == create_group
        process.terminate.assert_called_once_with()
        killpg.assert_not_called()

    def test_quiet_capture_retains_only_bounded_tail(self, tmp_path):
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "python").touch()
        payload = b"x" * (venv_module._CAPTURE_LIMIT_BYTES + 64) + b"useful-tail"
        process = _mock_process(returncode=1, stdout=payload)
        signals, killpg, _ = _group_signal_that_exits_on(signal.SIGTERM)
        with patch("muse.core.venv._OS_NAME", "posix"), \
             patch("muse.core.venv.os.getpgid", return_value=4242), \
             patch("muse.core.venv.os.getpgrp", return_value=31337), \
             patch("muse.core.venv.os.killpg", side_effect=killpg), \
             patch("muse.core.venv.subprocess.Popen", return_value=process):
            with pytest.raises(subprocess.CalledProcessError) as exc_info:
                install_into_venv(tmp_path, ["bogus"])

        captured = exc_info.value.stdout
        assert captured.startswith("[... ")
        assert "output bytes truncated" in captured
        assert captured.endswith("useful-tail")
        assert len(captured.encode()) <= venv_module._CAPTURE_LIMIT_BYTES + 80

    def test_public_owned_runner_can_return_nonzero_without_raising(self):
        process = _mock_process(returncode=9, stderr=b"failed safely")
        signals, killpg, _ = _group_signal_that_exits_on(signal.SIGTERM)
        with patch("muse.core.venv._OS_NAME", "posix"), \
             patch("muse.core.venv.os.getpgid", return_value=4242), \
             patch("muse.core.venv.os.getpgrp", return_value=31337), \
             patch("muse.core.venv.os.killpg", side_effect=killpg), \
             patch("muse.core.venv.subprocess.Popen", return_value=process):
            result = run_owned_command(
                ["fake-command"], timeout=1.0,
                capture_output=True, check=False,
            )

        assert result.returncode == 9
        assert result.stderr == "failed safely"
        assert signals == []

    def test_completed_leader_closes_stuck_capture_fd_without_group_signal(self):
        process = _mock_process(returncode=0)
        stream = MagicMock()
        thread = MagicMock()
        thread.name = "mock-stuck-reader"
        thread.is_alive.side_effect = [True, False]
        reader = venv_module._CaptureReader(
            stream=stream,
            capture=venv_module._BoundedCapture(),
            thread=thread,
        )

        with patch("muse.core.venv._OS_NAME", "posix"), \
             patch("muse.core.venv.os.getpgid", return_value=4242), \
             patch("muse.core.venv.os.getpgrp", return_value=31337), \
             patch("muse.core.venv.os.killpg") as killpg, \
             patch("muse.core.venv._start_capture", return_value=[reader]), \
             patch("muse.core.venv.subprocess.Popen", return_value=process):
            result = run_owned_command(
                ["fake-command"], timeout=1.0,
                capture_output=True, check=True,
            )

        assert result.returncode == 0
        killpg.assert_not_called()
        stream.close.assert_called_once_with()
        assert thread.join.call_count == 2

    @pytest.mark.skipif(
        os.name != "posix"
        or not hasattr(os, "fork")
        or not hasattr(os, "waitid")
        or not hasattr(os, "WNOWAIT"),
        reason="requires POSIX WNOWAIT",
    )
    def test_completed_leader_cleans_term_ignoring_descendant(self, tmp_path):
        """A child-held file lock proves the exact descendant released state."""
        ready_path = tmp_path / "owned-child.ready"
        lock_path = tmp_path / "owned-child.lock"
        control_path = tmp_path / "owned-child.sock"
        script = (
            "import fcntl, os, signal, socket, sys, time\n"
            "child = os.fork()\n"
            "if child == 0:\n"
            "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "    lock = open(sys.argv[2], 'w', encoding='utf-8')\n"
            "    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)\n"
            "    control = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
            "    control.bind(sys.argv[3])\n"
            "    control.listen(1)\n"
            "    control.settimeout(0.1)\n"
            "    with open(sys.argv[1], 'w', encoding='utf-8') as ready:\n"
            "        ready.write('ready')\n"
            "        ready.flush()\n"
            "        os.fsync(ready.fileno())\n"
            "    deadline = time.monotonic() + 10.0\n"
            "    while time.monotonic() < deadline:\n"
            "        try:\n"
            "            connection, _ = control.accept()\n"
            "        except TimeoutError:\n"
            "            continue\n"
            "        with connection:\n"
            "            if connection.recv(16) == b'stop':\n"
            "                break\n"
            "    os._exit(0)\n"
            "deadline = time.monotonic() + 3.0\n"
            "while not os.path.exists(sys.argv[1]) and time.monotonic() < deadline:\n"
            "    time.sleep(0.01)\n"
            "time.sleep(0.5)\n"
        )
        outcome = {}

        def invoke():
            try:
                outcome["result"] = run_owned_command(
                    [
                        sys.executable, "-c", script, str(ready_path),
                        str(lock_path), str(control_path),
                    ],
                    timeout=5.0,
                    capture_output=True,
                    check=True,
                )
            except BaseException as exc:
                outcome["error"] = exc

        runner = threading.Thread(target=invoke, daemon=True)
        runner.start()
        deadline = time.monotonic() + 4.0
        while not ready_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready_path.exists(), outcome

        probe = lock_path.open("w", encoding="utf-8")
        lock_acquired = False
        try:
            runner.join(timeout=6.0)
            assert not runner.is_alive()
            assert "error" not in outcome
            assert outcome["result"].returncode == 0
            try:
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_acquired = True
            except BlockingIOError:
                pass
            assert lock_acquired, "the exact owned descendant retained its lock"
        finally:
            if not lock_acquired:
                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                        client.settimeout(1.0)
                        client.connect(str(control_path))
                        client.sendall(b"stop")
                except (FileNotFoundError, ConnectionError, TimeoutError):
                    pass
            probe.close()
            runner.join(timeout=2.0)


class TestFindFreePort:
    def test_returns_an_int_in_range(self):
        p = find_free_port(start=9001, end=9999)
        assert 9001 <= p <= 9999

    def test_skips_bound_ports(self):
        import socket
        # Use OS-assigned ephemeral port (avoids flakiness on busy hosts).
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        bound_port = s.getsockname()[1]
        s.listen(1)
        try:
            p = find_free_port(start=bound_port, end=bound_port + 2)
            assert p != bound_port
        finally:
            s.close()

    def test_raises_when_no_free_port_in_range(self):
        import socket
        # Bind 3 consecutive ephemeral ports, then call find_free_port over
        # exactly that range so every port is taken.
        sockets = []
        try:
            # Bind first port; OS picks the number
            s0 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s0.bind(("127.0.0.1", 0))
            s0.listen(1)
            sockets.append(s0)
            port_start = s0.getsockname()[1]

            # Try to bind the next two sequential ports. If either is already
            # taken by some other process, adjust bound_ports accordingly so
            # the range we pass to find_free_port contains ONLY bound ports.
            bound_ports = [port_start]
            for offset in (1, 2):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    s.bind(("127.0.0.1", port_start + offset))
                    s.listen(1)
                    sockets.append(s)
                    bound_ports.append(port_start + offset)
                except OSError:
                    s.close()
                    break

            # Exhaust exactly the contiguous bound range
            with pytest.raises(RuntimeError, match="no free port"):
                find_free_port(start=bound_ports[0], end=bound_ports[-1])
        finally:
            for s in sockets:
                s.close()
