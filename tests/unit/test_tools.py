import pytest
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import subprocess as sp

from rimeX.tools import check_call, cdo, dir_is_empty, Timer


class TestCheckCall:
    """Tests for the check_call function"""

    def test_check_call_success(self):
        """Test successful command execution"""
        with patch('subprocess.check_call', return_value=0) as mock_check_call:
            result = check_call("echo 'hello'")
            assert result == 0
            mock_check_call.assert_called_once_with(
                "echo 'hello'",
                shell=True,
                env=None
            )

    def test_check_call_with_env(self):
        """Test check_call with environment variables"""
        test_env = {"TEST_VAR": "test_value"}
        with patch('subprocess.check_call', return_value=0) as mock_check_call:
            result = check_call("some_command", env=test_env)
            assert result == 0
            
            # Verify that env was merged with os.environ
            call_args = mock_check_call.call_args
            merged_env = call_args[1]['env']
            assert "TEST_VAR" in merged_env
            assert merged_env["TEST_VAR"] == "test_value"

    def test_check_call_dry_run(self):
        """Test dry_run mode doesn't execute subprocess"""
        with patch('subprocess.check_call') as mock_check_call:
            result = check_call("echo 'hello'", dry_run=True)
            assert result == 0
            # Verify subprocess.check_call was NOT called
            mock_check_call.assert_not_called()

    def test_check_call_prints_command(self, capsys):
        """Test that the command is printed before execution"""
        with patch('subprocess.check_call', return_value=0):
            check_call("test_command")
            captured = capsys.readouterr()
            assert "test_command" in captured.out

    def test_check_call_prints_even_on_dry_run(self, capsys):
        """Test that command is printed even on dry_run"""
        check_call("test_command", dry_run=True)
        captured = capsys.readouterr()
        assert "test_command" in captured.out

    def test_check_call_failure(self):
        """Test that subprocess errors are propagated"""
        with patch('subprocess.check_call', side_effect=sp.CalledProcessError(1, "cmd")):
            with pytest.raises(sp.CalledProcessError):
                check_call("failing_command")

    def test_check_call_with_env_and_dry_run(self):
        """Test env is not used when dry_run is True"""
        test_env = {"TEST_VAR": "test_value"}
        with patch('subprocess.check_call') as mock_check_call:
            result = check_call("cmd", env=test_env, dry_run=True)
            assert result == 0
            mock_check_call.assert_not_called()


class TestCdo:
    """Tests for the cdo function"""

    def test_cdo_basic(self):
        """Test basic cdo command construction"""
        with patch('rimeX.tools.check_call', return_value=0) as mock_check_call:
            result = cdo("seltimestep,1 input.nc output.nc")
            assert result == 0
            # Verify 'cdo' prefix was added
            mock_check_call.assert_called_once()
            cmd = mock_check_call.call_args[0][0]
            assert cmd.startswith("cdo ")

    def test_cdo_with_dry_run(self):
        """Test cdo respects dry_run parameter"""
        with patch('rimeX.tools.check_call', return_value=0) as mock_check_call:
            cdo("some_operation", dry_run=True)
            mock_check_call.assert_called_once()
            # Verify dry_run was passed through
            assert mock_check_call.call_args[1]['dry_run'] == True

    def test_cdo_with_env(self):
        """Test cdo passes env parameter to check_call"""
        test_env = {"CDO_VAR": "value"}
        with patch('rimeX.tools.check_call', return_value=0) as mock_check_call:
            cdo("operation", env=test_env)
            mock_check_call.assert_called_once()
            assert mock_check_call.call_args[1]['env'] == test_env

    def test_cdo_command_format(self):
        """Test that cdo command is properly formatted"""
        with patch('rimeX.tools.check_call', return_value=0) as mock_check_call:
            test_op = "mergetime input1.nc input2.nc output.nc"
            cdo(test_op)
            cmd = mock_check_call.call_args[0][0]
            assert cmd == f"cdo {test_op}"


class TestDirIsEmpty:
    """Tests for the dir_is_empty function"""

    def test_empty_directory(self):
        """Test with an empty directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert dir_is_empty(tmpdir) is True

    def test_directory_with_file(self):
        """Test with directory containing a file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file in the directory
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test content")
            assert dir_is_empty(tmpdir) is False

    def test_directory_with_subdirectory(self):
        """Test with directory containing a subdirectory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a subdirectory
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            assert dir_is_empty(tmpdir) is False

    def test_directory_with_hidden_file(self):
        """Test with directory containing a hidden file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a hidden file
            hidden_file = Path(tmpdir) / ".hidden"
            hidden_file.write_text("hidden content")
            assert dir_is_empty(tmpdir) is False

    def test_multiple_files(self):
        """Test with directory containing multiple files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file1.txt").write_text("content1")
            Path(tmpdir, "file2.txt").write_text("content2")
            assert dir_is_empty(tmpdir) is False


class TestTimer:
    """Tests for the Timer class"""

    def test_timer_initialization(self):
        """Test Timer initializes correctly"""
        timer = Timer()
        assert timer.total == 0
        assert hasattr(timer, 't')

    def test_timer_reset(self):
        """Test Timer.reset() updates the time"""
        timer = Timer()
        initial_t = timer.t
        time.sleep(0.01)  # Small delay
        timer.reset()
        assert timer.t > initial_t

    def test_timer_check_timing(self):
        """Test Timer.check() measures elapsed time"""
        timer = Timer()
        timer.reset()
        time.sleep(0.05)  # Sleep for 50ms
        delta = timer._check()
        # Allow some tolerance (should be close to 50ms)
        assert 40 < delta * 1000 < 100  # Convert to ms

    def test_timer_check_accumulates(self):
        """Test Timer.check() accumulates in total"""
        timer = Timer()
        timer.reset()
        time.sleep(0.02)
        timer._check()
        first_total = timer.total
        
        timer.reset()
        time.sleep(0.02)
        timer._check()
        second_total = timer.total
        
        # Total should increase
        assert second_total > first_total

    def test_timer_check_message(self, capsys):
        """Test Timer.check() prints message and time"""
        timer = Timer()
        timer.reset()
        time.sleep(0.01)
        timer.check("Test operation")
        
        captured = capsys.readouterr()
        assert "Test operation" in captured.out
        assert "ms" in captured.out

    def test_timer_check_resets_after_call(self):
        """Test Timer.check() resets the timer"""
        timer = Timer()
        timer.reset()
        first_t = timer.t
        
        time.sleep(0.01)
        timer.check("first")
        first_after_check = timer.t
        
        # Timer should be reset (t should be updated)
        assert first_after_check > first_t

    def test_timer_stop_message(self, capsys):
        """Test Timer.stop() prints total time"""
        timer = Timer()
        timer.reset()
        time.sleep(0.01)
        timer.stop("Total time")
        
        captured = capsys.readouterr()
        assert "Total time" in captured.out
        assert "ms" in captured.out

    def test_timer_stop_resets_total(self):
        """Test Timer.stop() resets total to 0"""
        timer = Timer()
        timer.reset()
        time.sleep(0.01)
        timer._check()
        assert timer.total > 0
        
        timer.stop("test")
        assert timer.total == 0

    def test_timer_multiple_checks(self):
        """Test multiple check() calls accumulate correctly"""
        timer = Timer()
        total_expected = 0
        
        for _ in range(3):
            timer.reset()
            time.sleep(0.01)
            delta = timer._check()
            total_expected += delta
        
        # Total should roughly match sum of individual delays
        assert timer.total > 0
        assert abs(timer.total - total_expected) < 0.01  # Some tolerance

    def test_timer_stop_without_check(self, capsys):
        """Test Timer.stop() works without prior check()"""
        timer = Timer()
        timer.reset()
        time.sleep(0.01)
        timer.stop("test")
        
        captured = capsys.readouterr()
        assert "test" in captured.out

    def test_timer_precision(self):
        """Test Timer measures time with reasonable precision"""
        timer = Timer()
        timer.reset()
        time.sleep(0.05)
        delta = timer._check()
        
        # Should be approximately 50ms (with tolerance)
        delta_ms = delta * 1000
        assert 40 < delta_ms < 100


class TestTimerIntegration:
    """Integration tests for Timer class"""

    def test_timer_full_workflow(self, capsys):
        """Test complete Timer workflow"""
        timer = Timer()
        
        # First operation
        timer.reset()
        time.sleep(0.01)
        timer.check("Operation 1")
        
        # Second operation
        timer.reset()
        time.sleep(0.01)
        timer.check("Operation 2")
        
        # Final measurement
        timer.stop("Total operations")
        
        captured = capsys.readouterr()
        assert "Operation 1" in captured.out
        assert "Operation 2" in captured.out
        assert "Total operations" in captured.out

    def test_timer_without_reset(self):
        """Test Timer behavior when not explicitly reset"""
        timer = Timer()
        # Timer initializes with current time in __init__
        time.sleep(0.01)
        delta = timer._check()
        assert delta > 0

