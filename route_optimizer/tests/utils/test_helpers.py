import unittest
import numpy as np
import datetime
import json
from unittest.mock import patch

from route_optimizer.utils.helpers import (
    convert_minutes_to_time_str,
    convert_time_str_to_minutes,
    format_route_for_display,
    apply_external_factors,
    detect_isolated_nodes,
    safe_json_dumps,
    format_duration
)
from route_optimizer.core.constants import TIME_SCALING_FACTOR # Used by format_duration

class TestHelpers(unittest.TestCase):

    def test_convert_minutes_to_time_str(self):
        self.assertEqual(convert_minutes_to_time_str(0), "00:00")
        self.assertEqual(convert_minutes_to_time_str(570), "09:30")  # 9:30 AM
        self.assertEqual(convert_minutes_to_time_str(825), "13:45")  # 1:45 PM
        self.assertEqual(convert_minutes_to_time_str(1439), "23:59") # 11:59 PM
        self.assertEqual(convert_minutes_to_time_str(1500), "25:00") # More than 24 hours

    def test_convert_time_str_to_minutes(self):
        self.assertEqual(convert_time_str_to_minutes("00:00"), 0)
        self.assertEqual(convert_time_str_to_minutes("09:30"), 570)
        self.assertEqual(convert_time_str_to_minutes("13:45"), 825)
        self.assertEqual(convert_time_str_to_minutes("23:59"), 1439)

        # Test invalid formats
        with self.assertLogs('route_optimizer.utils.helpers', level='ERROR') as cm:
            self.assertEqual(convert_time_str_to_minutes("9:30"), 0) # Invalid, expects HH:MM
            self.assertIn("Invalid time string format: 9:30", cm.output[0])
        
        with self.assertLogs('route_optimizer.utils.helpers', level='ERROR') as cm:
            self.assertEqual(convert_time_str_to_minutes("0930"), 0)
            self.assertIn("Invalid time string format: 0930", cm.output[0])

        with self.assertLogs('route_optimizer.utils.helpers', level='ERROR') as cm:
            self.assertEqual(convert_time_str_to_minutes("abc"), 0)
            self.assertIn("Invalid time string format: abc", cm.output[0])
        
        with self.assertLogs('route_optimizer.utils.helpers', level='ERROR') as cm:
            self.assertEqual(convert_time_str_to_minutes(None), 0) # type: ignore
            self.assertIn("Invalid time string format: None", cm.output[0])


    def test_format_route_for_display(self):
        location_names = {"L1": "Depot", "L2": "Customer A", "L3": "Customer B"}
        self.assertEqual(format_route_for_display([], location_names), "")
        self.assertEqual(format_route_for_display(["L1"], location_names), "Depot")
        self.assertEqual(
            format_route_for_display(["L1", "L2", "L3"], location_names),
            "Depot → Customer A → Customer B"
        )
        self.assertEqual(
            format_route_for_display(["L1", "L4", "L2"], location_names), # L4 not in names
            "Depot → L4 → Customer A"
        )

    def test_apply_external_factors(self):
        dist_matrix = np.array([[0, 10], [10, 0]], dtype=float)
        time_matrix = np.array([[0, 20], [20, 0]], dtype=float)
        
        factors = {(0, 1): 1.5} # Time from 0 to 1 should be 20 * 1.5 = 30
        
        updated_dist, updated_time = apply_external_factors(dist_matrix, time_matrix, factors)
        
        # Check originals are not modified
        self.assertTrue(np.array_equal(dist_matrix, np.array([[0, 10], [10, 0]])))
        self.assertTrue(np.array_equal(time_matrix, np.array([[0, 20], [20, 0]])))
        
        # Check distance matrix is unchanged
        self.assertTrue(np.array_equal(updated_dist, dist_matrix))
        
        # Check time matrix is updated
        expected_time_matrix = np.array([[0, 30], [20, 0]], dtype=float)
        self.assertTrue(np.array_equal(updated_time, expected_time_matrix))

        # Test with empty factors
        updated_dist_no_factors, updated_time_no_factors = apply_external_factors(dist_matrix, time_matrix, {})
        self.assertTrue(np.array_equal(updated_dist_no_factors, dist_matrix))
        self.assertTrue(np.array_equal(updated_time_no_factors, time_matrix))

    def test_detect_isolated_nodes(self):
        graph1 = {"A": {"B": 1}, "B": {"A": 1, "C": 1}, "C": {"B": 1}} # No isolated
        self.assertEqual(detect_isolated_nodes(graph1), [])

        graph2 = {"A": {"B": 1}, "B": {"A": 1}, "C": {}} # C is isolated (no out, no in from A/B)
        self.assertEqual(sorted(detect_isolated_nodes(graph2)), sorted(["C"]))
        
        graph3 = {"A": {}, "B": {}, "C": {"A": 1}} # A, B are isolated
        self.assertEqual(sorted(detect_isolated_nodes(graph3)), sorted(["A", "B"]))

        graph4 = {"A": {"B": 1}, "B": {}, "C": {"B":1}} # B has incoming but no outgoing -> not isolated by this definition
        self.assertEqual(sorted(detect_isolated_nodes(graph4)), sorted(["B"]))

        graph5 = {"A": {}} # A is isolated
        self.assertEqual(detect_isolated_nodes(graph5), ["A"])
        
        graph6 = {} # Empty graph
        self.assertEqual(detect_isolated_nodes(graph6), [])
        
        graph7 = {"A": {"B":1}, "B": {"C":1}, "C": {"A":1}, "D":{}} # D is isolated
        self.assertEqual(detect_isolated_nodes(graph7), ["D"])

    def test_safe_json_dumps(self):
        # Basic types
        self.assertEqual(safe_json_dumps({"key": "value", "num": 1}), '{"key": "value", "num": 1}')
        self.assertEqual(safe_json_dumps([1, "two", None]), '[1, "two", null]')

        # Datetime
        dt = datetime.datetime(2023, 1, 1, 12, 30, 0)
        d = datetime.date(2023, 1, 1)
        self.assertEqual(safe_json_dumps(dt), f'"{dt.isoformat()}"')
        self.assertEqual(safe_json_dumps(d), f'"{d.isoformat()}"')

        # Numpy types
        arr = np.array([1, 2, 3])
        self.assertEqual(safe_json_dumps(arr), '[1, 2, 3]')
        np_int = np.int64(5)
        self.assertEqual(safe_json_dumps(np_int), '5')
        np_float = np.float64(5.5)
        self.assertEqual(safe_json_dumps(np_float), '5.5')

        # Custom object with __dict__
        class MyObject:
            def __init__(self, x):
                self.x = x
        my_obj = MyObject(10)
        self.assertEqual(safe_json_dumps(my_obj), '{"x": 10}')

        # Unserializable object (function)
        def my_func(): pass
        self.assertEqual(safe_json_dumps(my_func), f'"{str(my_func)}"')

    def test_format_duration(self):
        # Assuming TIME_SCALING_FACTOR is 60 (seconds per minute for the solver's internal minute representation)
        # The function `format_duration` expects seconds input and scales based on TIME_SCALING_FACTOR
        # divmod(seconds, 60 * TIME_SCALING_FACTOR) for hours -> divmod(seconds, 3600)
        # divmod(remainder, TIME_SCALING_FACTOR) for minutes -> divmod(remainder, 60)
        
        self.assertEqual(format_duration(0), "0m 0s") # Shows 0m 0s for zero
        self.assertEqual(format_duration(30), "0m 30s")
        self.assertEqual(format_duration(60), "1m 0s") # format_duration shows 0s if minutes are present
        self.assertEqual(format_duration(300), "5m 0s") # 5 minutes
        self.assertEqual(format_duration(330), "5m 30s") # 5 minutes 30 seconds
        
        self.assertEqual(format_duration(3600), "1h 0m 0s") # 1 hour
        self.assertEqual(format_duration(5400), "1h 30m 0s") # 1 hour 30 minutes
        self.assertEqual(format_duration(5415), "1h 30m 15s") # 1 hour 30 minutes 15 seconds

        self.assertEqual(format_duration(0.5), "0m 0s") # Sub-second, rounds down to 0s if no larger units
        self.assertEqual(format_duration(1.5), "0m 1s") # Sub-second, rounds down to 0s if no larger units
        
        # Test with a specific TIME_SCALING_FACTOR if it were different
        # For example, if TIME_SCALING_FACTOR was 1 (meaning solver works directly in seconds)
        # Then divmod(seconds, 60 * 1) and divmod(remainder, 1) would be different
        # However, the current structure implies TIME_SCALING_FACTOR=60 for "minutes" in solver,
        # leading to standard hour/minute/second conversion from total input seconds.

        # Edge case: small number of seconds
        self.assertEqual(format_duration(1), "0m 1s")
        # Edge case: just under a minute
        self.assertEqual(format_duration(59), "0m 59s")
        # Edge case: just under an hour
        self.assertEqual(format_duration(3599), "59m 59s")


if __name__ == '__main__':
    unittest.main()
