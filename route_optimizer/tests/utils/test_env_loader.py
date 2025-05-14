import unittest
import os
import tempfile
import logging
from route_optimizer.utils.env_loader import load_env_from_file

class TestEnvLoader(unittest.TestCase):

    def setUp(self):
        # Store original environment variables to restore them later
        self.original_environ = os.environ.copy()
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_env_file_path = os.path.join(self.temp_dir.name, ".env.test")

        # Keys that might be set by tests, to be cleaned up specifically
        self.test_keys_to_clear = [
            "TEST_KEY1", "TEST_KEY2", "TEST_KEY3", "EXISTING_KEY",
            "KEY_NO_VALUE", "  LEADING_SPACE_KEY", "TRAILING_SPACE_KEY  "
        ]

    def tearDown(self):
        # Clean up specific environment variables set by tests
        for key in self.test_keys_to_clear:
            if key in os.environ:
                del os.environ[key]
            if key.strip() in os.environ: # For keys that might have been stripped
                 del os.environ[key.strip()]


        # Restore original environment - this is a more robust way
        os.environ.clear()
        os.environ.update(self.original_environ)
        
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def create_test_env_file(self, content):
        with open(self.test_env_file_path, 'w') as f:
            f.write(content)

    def test_load_env_successful(self):
        content = (
            "TEST_KEY1=test_value1\n"
            "# This is a comment\n"
            "TEST_KEY2 = test_value2_with_spaces  \n"
            "  TEST_KEY3   =    test_value3_more_spaces\n"
            "\n" # Empty line
            "KEY_NO_VALUE=\n"
            "  LEADING_SPACE_KEY=leading_value\n"
            "TRAILING_SPACE_KEY  =trailing_value\n"
        )
        self.create_test_env_file(content)

        with self.assertLogs('route_optimizer.utils.env_loader', level='INFO') as cm:
            result = load_env_from_file(self.test_env_file_path)
        
        self.assertTrue(result)
        self.assertEqual(os.environ.get("TEST_KEY1"), "test_value1")
        self.assertEqual(os.environ.get("TEST_KEY2"), "test_value2_with_spaces")
        self.assertEqual(os.environ.get("TEST_KEY3"), "test_value3_more_spaces")
        self.assertEqual(os.environ.get("KEY_NO_VALUE"), "")
        self.assertEqual(os.environ.get("LEADING_SPACE_KEY"), "leading_value")
        self.assertEqual(os.environ.get("TRAILING_SPACE_KEY"), "trailing_value")
        
        self.assertIn(f"INFO:route_optimizer.utils.env_loader:Loaded environment variables from {self.test_env_file_path}", cm.output)

    def test_load_env_file_not_found(self):
        non_existent_file = os.path.join(self.temp_dir.name, "non_existent.env")
        with self.assertLogs('route_optimizer.utils.env_loader', level='WARNING') as cm:
            result = load_env_from_file(non_existent_file)
        
        self.assertFalse(result)
        self.assertIn(f"WARNING:route_optimizer.utils.env_loader:Environment file not found: {non_existent_file}", cm.output)

    def test_load_env_malformed_file_no_equals(self):
        self.create_test_env_file("MALFORMED_LINE_NO_EQUALS_SIGN")
        
        with self.assertLogs('route_optimizer.utils.env_loader', level='ERROR') as cm:
            result = load_env_from_file(self.test_env_file_path)
            
        self.assertFalse(result)
        self.assertIn(f"ERROR:route_optimizer.utils.env_loader:Error loading environment variables from {self.test_env_file_path}: not enough values to unpack (expected 2, got 1)", cm.output)

    def test_load_env_empty_key_after_strip(self):
        # Line like " =value " would result in an empty key after strip
        # os.environ does not allow empty string as a key
        self.create_test_env_file("=value_for_empty_key")
        
        with self.assertLogs('route_optimizer.utils.env_loader', level='ERROR') as cm:
            result = load_env_from_file(self.test_env_file_path)
            
        self.assertFalse(result)
        # The exact error message for empty key can vary by OS/Python version,
        # so we check for a general error log.
        self.assertTrue(any(f"ERROR:route_optimizer.utils.env_loader:Error loading environment variables from {self.test_env_file_path}" in log_msg for log_msg in cm.output))


    def test_load_env_overwrite_existing(self):
        os.environ["EXISTING_KEY"] = "original_value"
        self.create_test_env_file("EXISTING_KEY=new_value")
        
        result = load_env_from_file(self.test_env_file_path)
        
        self.assertTrue(result)
        self.assertEqual(os.environ.get("EXISTING_KEY"), "new_value")

    def test_load_env_empty_file(self):
        self.create_test_env_file("") # Empty file
        
        with self.assertLogs('route_optimizer.utils.env_loader', level='INFO') as cm:
            result = load_env_from_file(self.test_env_file_path)
            
        self.assertTrue(result)
        self.assertIn(f"INFO:route_optimizer.utils.env_loader:Loaded environment variables from {self.test_env_file_path}", cm.output)
        # No new specific keys should be set beyond what was originally there or standard system vars
        self.assertNotIn("TEST_KEY1", os.environ) 

    def test_load_env_file_with_only_comments_and_empty_lines(self):
        content = (
            "# This is a comment\n"
            "\n"
            "    # Another comment with leading spaces\n"
            "  \n"
        )
        self.create_test_env_file(content)
        
        # Store current env keys to check no new app-specific keys are added
        initial_env_keys = set(os.environ.keys())

        with self.assertLogs('route_optimizer.utils.env_loader', level='INFO') as cm:
            result = load_env_from_file(self.test_env_file_path)
            
        self.assertTrue(result)
        self.assertIn(f"INFO:route_optimizer.utils.env_loader:Loaded environment variables from {self.test_env_file_path}", cm.output)
        
        current_env_keys = set(os.environ.keys())
        newly_added_keys = current_env_keys - initial_env_keys
        # Assert that no unexpected keys (like from self.test_keys_to_clear) were added
        self.assertFalse(any(key in newly_added_keys for key in self.test_keys_to_clear if key not in initial_env_keys))


if __name__ == '__main__':
    unittest.main()
