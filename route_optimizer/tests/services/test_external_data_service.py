import unittest
from unittest.mock import patch, MagicMock, call
import random
import time # For time.sleep patching
import json

from requests.exceptions import RequestException, HTTPError
from requests import Response as RequestsResponse # To avoid confusion with DRF Response

from route_optimizer.core.types_1 import Location
from route_optimizer.services.external_data_service import ExternalDataService
# Assuming these are loaded from settings correctly during tests
from route_optimizer.settings import MAX_RETRIES, BACKOFF_FACTOR, RETRY_DELAY_SECONDS


class TestExternalDataServiceInitialization(unittest.TestCase):
    def test_initialization_defaults(self):
        service = ExternalDataService()
        self.assertIsNone(service.traffic_api_key)
        self.assertIsNone(service.weather_api_key)
        self.assertFalse(service.use_mocks)
        self.assertEqual(service.traffic_api_url, "https_traffic_api_example_com_v1_data")
        self.assertEqual(service.weather_api_url, "https_weather_api_example_com_v1_current")
        self.assertEqual(service.roadblock_api_url, "https_roadblock_api_example_com_v1_alerts")

    def test_initialization_with_params(self):
        service = ExternalDataService(
            traffic_api_key="traffic_key",
            weather_api_key="weather_key",
            use_mocks=True
        )
        self.assertEqual(service.traffic_api_key, "traffic_key")
        self.assertEqual(service.weather_api_key, "weather_key")
        self.assertTrue(service.use_mocks)


class TestMakeApiRequest(unittest.TestCase):
    def setUp(self):
        self.service = ExternalDataService()
        self.test_url = "http://fakeapi.com/data"
        self.test_params = {"param1": "value1"}

    @patch('requests.get')
    def test_make_api_request_success(self, mock_requests_get):
        mock_response = MagicMock(spec=RequestsResponse)
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success", "data": "some_data"}
        mock_requests_get.return_value = mock_response

        result = self.service._make_api_request(self.test_url, self.test_params, api_key="test_key")

        self.assertEqual(result, {"status": "success", "data": "some_data"})
        mock_requests_get.assert_called_once_with(
            self.test_url,
            params=self.test_params,
            headers={'Authorization': 'Bearer test_key'},
            timeout=10
        )
        mock_response.raise_for_status.assert_called_once()

    @patch('requests.get')
    def test_make_api_request_http_error_429_retry_success(self, mock_requests_get):
        mock_response_fail = MagicMock(spec=RequestsResponse)
        mock_response_fail.status_code = 429 # Rate limit
        http_error = HTTPError(response=mock_response_fail)
        mock_response_fail.raise_for_status.side_effect = http_error # First call raises

        mock_response_success = MagicMock(spec=RequestsResponse)
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"status": "success"}
        
        # First call fails, second succeeds
        mock_requests_get.side_effect = [mock_response_fail, mock_response_success]

        with patch('time.sleep') as mock_sleep:
            result = self.service._make_api_request(self.test_url, self.test_params)
            self.assertEqual(result, {"status": "success"})
            self.assertEqual(mock_requests_get.call_count, 2)
            mock_sleep.assert_called_once_with(RETRY_DELAY_SECONDS * (BACKOFF_FACTOR ** 0))

    @patch('requests.get')
    def test_make_api_request_http_error_401_no_retry(self, mock_requests_get):
        mock_response_fail = MagicMock(spec=RequestsResponse)
        mock_response_fail.status_code = 401 # Auth error
        http_error = HTTPError(response=mock_response_fail)
        mock_response_fail.raise_for_status.side_effect = http_error
        mock_requests_get.return_value = mock_response_fail

        with patch('time.sleep') as mock_sleep:
            result = self.service._make_api_request(self.test_url, self.test_params)
            self.assertIsNone(result)
            mock_requests_get.assert_called_once() # Should not retry
            mock_sleep.assert_not_called()

    @patch('requests.get')
    def test_make_api_request_request_exception_retry_fail(self, mock_requests_get):
        mock_requests_get.side_effect = RequestException("Connection failed")

        with patch('time.sleep') as mock_sleep:
            result = self.service._make_api_request(self.test_url, self.test_params)
            self.assertIsNone(result)
            self.assertEqual(mock_requests_get.call_count, MAX_RETRIES)
            self.assertEqual(mock_sleep.call_count, MAX_RETRIES - 1)

    @patch('requests.get')
    def test_make_api_request_json_decode_error(self, mock_requests_get):
        mock_response = MagicMock(spec=RequestsResponse)
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("msg", "doc", 0)
        mock_requests_get.return_value = mock_response

        result = self.service._make_api_request(self.test_url, self.test_params)
        self.assertIsNone(result)
        mock_requests_get.assert_called_once()


class TestExternalDataServiceMethods(unittest.TestCase):
    def setUp(self):
        self.locations = [
            Location(id="loc1", latitude=1.0, longitude=1.0, name="Loc1"),
            Location(id="loc2", latitude=2.0, longitude=2.0, name="Loc2"),
            Location(id="loc3", latitude=3.0, longitude=3.0, name="Loc3"),
        ]

    # --- Test get_traffic_data ---
    def test_get_traffic_data_use_mocks(self):
        service = ExternalDataService(use_mocks=True)
        with patch.object(service, '_mock_traffic_data', return_value={"mocked": True}) as mock_method:
            result = service.get_traffic_data(self.locations)
            self.assertEqual(result, {"mocked": True})
            mock_method.assert_called_once_with(self.locations)

    def test_get_traffic_data_no_api_key_fallback_to_mock(self):
        service = ExternalDataService(traffic_api_key=None, use_mocks=False)
        with patch.object(service, '_mock_traffic_data', return_value={"mocked_fallback": True}) as mock_method:
            result = service.get_traffic_data(self.locations)
            self.assertEqual(result, {"mocked_fallback": True})
            mock_method.assert_called_once_with(self.locations)

    @patch.object(ExternalDataService, '_make_api_request')
    def test_get_traffic_data_api_success(self, mock_make_request):
        service = ExternalDataService(traffic_api_key="fake_key", use_mocks=False)
        api_response = {
            "status": "success",
            "traffic_factors": [
                {"from_idx": 0, "to_idx": 1, "factor": 1.5},
                {"from_idx": 1, "to_idx": 2, "factor": 1.2}
            ]
        }
        mock_make_request.return_value = api_response
        
        result = service.get_traffic_data(self.locations)
        
        expected_result = {(0, 1): 1.5, (1, 2): 1.2}
        self.assertEqual(result, expected_result)
        mock_make_request.assert_called_once_with(
            service.traffic_api_url,
            {'location_ids': 'loc1,loc2,loc3'},
            "fake_key"
        )

    @patch.object(ExternalDataService, '_make_api_request')
    def test_get_traffic_data_api_fail_fallback_to_mock(self, mock_make_request):
        service = ExternalDataService(traffic_api_key="fake_key", use_mocks=False)
        mock_make_request.return_value = None # API call fails
        with patch.object(service, '_mock_traffic_data', return_value={"mock_on_api_fail": True}) as mock_fallback:
            result = service.get_traffic_data(self.locations)
            self.assertEqual(result, {"mock_on_api_fail": True})
            mock_fallback.assert_called_once_with(self.locations)

    # --- Test get_weather_data ---
    def test_get_weather_data_use_mocks(self):
        service = ExternalDataService(use_mocks=True)
        with patch.object(service, '_mock_weather_data', return_value={"loc1": {"mocked": True}}) as mock_method:
            result = service.get_weather_data(self.locations)
            self.assertEqual(result, {"loc1": {"mocked": True}})
            mock_method.assert_called_once_with(self.locations)

    @patch.object(ExternalDataService, '_make_api_request')
    def test_get_weather_data_api_success_all_locations(self, mock_make_request):
        service = ExternalDataService(weather_api_key="fake_key", use_mocks=False)
        
        # Simulate successful API responses for each location
        def side_effect_weather(url, params, api_key):
            loc_id_map = {"1.0": "loc1", "2.0": "loc2", "3.0": "loc3"} # Based on lat
            loc_id = loc_id_map[str(params['lat'])]
            return {
                "status": "success",
                "weather": {"condition": "Sunny", "temperature_celsius": 25, "impact_factor": 1.0, "loc_id_debug": loc_id}
            }
        mock_make_request.side_effect = side_effect_weather
        
        result = service.get_weather_data(self.locations)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result["loc1"]["condition"], "Sunny")
        self.assertEqual(result["loc2"]["impact_factor"], 1.0)
        self.assertEqual(mock_make_request.call_count, len(self.locations))

    @patch.object(ExternalDataService, '_make_api_request')
    def test_get_weather_data_api_partial_fail_fallback_for_failed(self, mock_make_request):
        service = ExternalDataService(weather_api_key="fake_key", use_mocks=False)
        
        def side_effect_weather_partial(url, params, api_key):
            if params['lat'] == 1.0: # loc1 success
                return {"status": "success", "weather": {"condition": "Cloudy", "temperature_celsius": 15}}
            return None # Other locations fail
        mock_make_request.side_effect = side_effect_weather_partial
        
        # Mock the fallback for individual locations
        with patch.object(service, '_mock_weather_data') as mock_fallback_method:
            # Configure mock_fallback_method to return specific data for specific calls
            def mock_weather_data_side_effect(locations_arg):
                if len(locations_arg) == 1 and locations_arg[0].id == "loc2":
                    return {"loc2": {"condition": "MockRain", "temperature": 10, "impact_factor": 1.2}}
                if len(locations_arg) == 1 and locations_arg[0].id == "loc3":
                     return {"loc3": {"condition": "MockSnow", "temperature": 0, "impact_factor": 1.5}}
                return {} # Default empty for unexpected calls
            mock_fallback_method.side_effect = mock_weather_data_side_effect

            result = service.get_weather_data(self.locations)

        self.assertEqual(len(result), 3)
        self.assertEqual(result["loc1"]["condition"], "Cloudy")
        self.assertEqual(result["loc2"]["condition"], "MockRain")
        self.assertEqual(result["loc3"]["condition"], "MockSnow")
        self.assertEqual(mock_make_request.call_count, len(self.locations))
        # Check _mock_weather_data was called for loc2 and loc3
        self.assertIn(call([self.locations[1]]), mock_fallback_method.call_args_list)
        self.assertIn(call([self.locations[2]]), mock_fallback_method.call_args_list)


    # --- Test get_roadblock_data ---
    def test_get_roadblock_data_use_mocks(self):
        service = ExternalDataService(use_mocks=True)
        with patch.object(service, '_mock_roadblock_data', return_value=[("L1", "L2")]) as mock_method:
            result = service.get_roadblock_data(self.locations)
            self.assertEqual(result, [("L1", "L2")])
            mock_method.assert_called_once_with(self.locations)

    @patch.object(ExternalDataService, '_make_api_request')
    def test_get_roadblock_data_api_success(self, mock_make_request):
        service = ExternalDataService(use_mocks=False) # No API key needed for roadblocks in example
        api_response = {
            "status": "success",
            "roadblocks": [
                {"from_location_id": "loc1", "to_location_id": "loc2"},
            ]
        }
        mock_make_request.return_value = api_response
        
        result = service.get_roadblock_data(self.locations)
        
        expected_result = [("loc1", "loc2")]
        self.assertEqual(result, expected_result)
        
        # Check bbox (optional, depends on how strict you want to be with mock API params)
        min_lat = min(loc.latitude for loc in self.locations)
        max_lat = max(loc.latitude for loc in self.locations)
        min_lon = min(loc.longitude for loc in self.locations)
        max_lon = max(loc.longitude for loc in self.locations)
        expected_bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
        
        mock_make_request.assert_called_once_with(
            service.roadblock_api_url,
            {"bbox": expected_bbox},
            None 
        )

    # --- Test Mock Data Generators ---
    def test_mock_traffic_data(self):
        service = ExternalDataService()
        result = service._mock_traffic_data(self.locations)
        self.assertIsInstance(result, dict)
        for (from_idx, to_idx), factor in result.items():
            self.assertIsInstance(from_idx, int)
            self.assertIsInstance(to_idx, int)
            self.assertIsInstance(factor, float)
            self.assertGreaterEqual(factor, 1.0)
            self.assertNotEqual(from_idx, to_idx)

    def test_mock_weather_data(self):
        service = ExternalDataService()
        result = service._mock_weather_data(self.locations)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), len(self.locations))
        for loc_id, data in result.items():
            self.assertIn('condition', data)
            self.assertIn('temperature', data)
            self.assertIn('impact_factor', data)

    def test_mock_roadblock_data(self):
        service = ExternalDataService()
        result = service._mock_roadblock_data(self.locations)
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str)
            self.assertIsInstance(item[1], str)
            
    def test_mock_data_with_empty_locations(self):
        service = ExternalDataService()
        empty_locs = []
        self.assertEqual(service._mock_traffic_data(empty_locs), {})
        self.assertEqual(service._mock_weather_data(empty_locs), {})
        self.assertEqual(service._mock_roadblock_data(empty_locs), [])

    def test_mock_data_with_one_location(self):
        service = ExternalDataService()
        one_loc = [self.locations[0]]
        self.assertEqual(service._mock_traffic_data(one_loc), {})
        self.assertEqual(len(service._mock_weather_data(one_loc)), 1)
        self.assertEqual(service._mock_roadblock_data(one_loc), [])


    # --- Test Processing Methods ---
    def test_calculate_weather_impact(self):
        service = ExternalDataService()
        weather_data = {
            "loc1": {"impact_factor": 1.2},
            "loc2": {"impact_factor": 1.5},
            "loc3": {"impact_factor": 1.0} # No impact
        }
        result = service.calculate_weather_impact(weather_data, self.locations)
        # Expected: loc1-loc2 (max(1.2,1.5)=1.5), loc1-loc3 (max(1.2,1.0)=1.2), loc2-loc3 (max(1.5,1.0)=1.5)
        # Indices: loc1=0, loc2=1, loc3=2
        self.assertEqual(result.get((0,1)), 1.5) # loc1 -> loc2
        self.assertEqual(result.get((1,0)), 1.5) # loc2 -> loc1 (should be symmetrical based on max)
        self.assertEqual(result.get((0,2)), 1.2) # loc1 -> loc3
        self.assertIsNone(result.get((0,0))) # No self-loops

    def test_combine_traffic_and_weather(self):
        service = ExternalDataService()
        traffic = {(0,1): 1.2, (1,2): 1.1}
        weather = {(0,1): 1.5, (0,2): 1.3} # (0,2) is new, (0,1) overlaps
        
        result = service.combine_traffic_and_weather(traffic, weather)
        
        self.assertAlmostEqual(result[(0,1)], 1.2 * 1.5)
        self.assertEqual(result[(1,2)], 1.1) # Unchanged by weather
        self.assertEqual(result[(0,2)], 1.3) # Added by weather


if __name__ == '__main__':
    unittest.main()

