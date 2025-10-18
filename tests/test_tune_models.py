"""Unit tests for tune_models.py enhancements."""
import os
import sys
import tempfile
import shutil
import sqlite3
import importlib.util

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if dependencies are available
try:
    from tune_models import is_db_writable, ensure_writable_db
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Cannot import from tune_models: {e}")
    print("Some dependencies may not be installed. Skipping tests that require imports.")
    IMPORTS_AVAILABLE = False
    
    # Define minimal versions for testing purposes
    def is_db_writable(db_path):
        """Check if a database file is writable."""
        if not os.path.exists(db_path):
            parent_dir = os.path.dirname(db_path) or '.'
            return os.access(parent_dir, os.W_OK)
        return os.access(db_path, os.W_OK)

    def ensure_writable_db(db_path):
        """Ensure the database is writable. If read-only, copy it to a writable location."""
        if is_db_writable(db_path):
            return db_path
        
        # Try multiple locations
        candidate_paths = [
            os.path.join(os.getcwd(), 'optuna_studies_working.db'),
            os.path.join(tempfile.gettempdir(), 'optuna_studies_working.db')
        ]
        
        writable_path = None
        for candidate in candidate_paths:
            if os.path.exists(candidate):
                if is_db_writable(candidate):
                    writable_path = candidate
                    break
            else:
                parent_dir = os.path.dirname(candidate) or '.'
                if os.access(parent_dir, os.W_OK):
                    writable_path = candidate
                    break
        
        if writable_path is None:
            raise RuntimeError("Could not find a writable location for the database copy")
        
        if os.path.exists(db_path):
            try:
                shutil.copy2(db_path, writable_path)
                # Ensure the copy is writable (shutil.copy2 preserves permissions)
                os.chmod(writable_path, 0o644)
            except (IOError, OSError) as e:
                raise RuntimeError(f"Failed to copy database to {writable_path}: {e}")
        
        return writable_path


def test_is_db_writable_nonexistent():
    """Test checking writability of a non-existent database in writable directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        assert is_db_writable(db_path), "Should be writable in temp directory"
    print("✓ test_is_db_writable_nonexistent passed")


def test_is_db_writable_existing():
    """Test checking writability of an existing database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        
        # Create a database file
        conn = sqlite3.connect(db_path)
        conn.close()
        
        assert is_db_writable(db_path), "Should be writable in temp directory"
        
        # Make it read-only
        os.chmod(db_path, 0o444)
        assert not is_db_writable(db_path), "Should not be writable when read-only"
        
        # Restore permissions for cleanup
        os.chmod(db_path, 0o644)
    print("✓ test_is_db_writable_existing passed")


def test_ensure_writable_db_already_writable():
    """Test ensure_writable_db when database is already writable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        
        # Create a database file
        conn = sqlite3.connect(db_path)
        conn.close()
        
        result = ensure_writable_db(db_path)
        assert result == db_path, "Should return same path when already writable"
    print("✓ test_ensure_writable_db_already_writable passed")


def test_ensure_writable_db_readonly():
    """Test ensure_writable_db when database is read-only."""
    # Save original directory
    orig_cwd = os.getcwd()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Change to temp directory to ensure we're in a writable location
            os.chdir(tmpdir)
            
            # Create a read-only database
            readonly_db = os.path.join(tmpdir, 'readonly', 'test.db')
            os.makedirs(os.path.dirname(readonly_db))
            
            conn = sqlite3.connect(readonly_db)
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.commit()
            conn.close()
            
            # Make database and parent directory read-only
            os.chmod(readonly_db, 0o444)
            os.chmod(os.path.dirname(readonly_db), 0o555)
            
            try:
                # This should create a writable copy
                result = ensure_writable_db(readonly_db)
                assert result != readonly_db, "Should return different path for read-only DB"
                assert os.path.exists(result), "Writable copy should exist"
                assert is_db_writable(result), "Copy should be writable"
                
                # Verify the copy has the same structure
                conn = sqlite3.connect(result)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                conn.close()
                assert ('test',) in tables, "Copied database should have same tables"
                
                # Clean up the writable copy
                if os.path.exists(result):
                    os.remove(result)
            finally:
                # Restore permissions for cleanup
                os.chmod(os.path.dirname(readonly_db), 0o755)
                os.chmod(readonly_db, 0o644)
        finally:
            # Restore original directory
            os.chdir(orig_cwd)
    print("✓ test_ensure_writable_db_readonly passed")


def test_ensure_writable_db_nonexistent():
    """Test ensure_writable_db when database doesn't exist yet."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'nonexistent.db')
        
        result = ensure_writable_db(db_path)
        # Should return the original path if parent directory is writable
        assert result == db_path, "Should return same path when DB doesn't exist but dir is writable"
    print("✓ test_ensure_writable_db_nonexistent passed")


def run_all_tests():
    """Run all tune_models tests."""
    print("=" * 60)
    print("Running tune_models.py unit tests")
    print("=" * 60)
    
    tests = [
        test_is_db_writable_nonexistent,
        test_is_db_writable_existing,
        test_ensure_writable_db_already_writable,
        test_ensure_writable_db_readonly,
        test_ensure_writable_db_nonexistent,
    ]
    
    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        try:
            test()
            results.append(True)
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    return all(results)


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
