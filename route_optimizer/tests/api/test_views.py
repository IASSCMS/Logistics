from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase, APIClient
from unittest.mock import patch, MagicMock

from route_optimizer.core.types_1 import OptimizationResult, Location
from route_optimizer.models import Vehicle, Delivery # Assuming these are dataclasses

class OptimizeRoutesViewTests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.optimize_url = reverse('optimize_routes_create') # Matches operation_id in OptimizeRoutesView

        # Sample data for requests
        self.location_data1 = {"id": "depot", "name": "Depot XYZ", "latitude": 34.0522, "longitude": -118.2437, "is_depot": True}
        self.location_data2 = {"id": "customer1", "name": "Customer Alpha", "latitude": 34.0523, "longitude": -118.2438}
        self.vehicle_data1 = {"id": "vehicle1", "capacity": 100.0, "start_location_id": "depot"}
        self.delivery_data1 = {"id": "delivery1", "location_id": "customer1", "demand": 10.0}

        self.valid_request_data = {
            "locations": [self.location_data1, self.location_data2],
            "vehicles": [self.vehicle_data1],
            "deliveries": [self.delivery_data1],
            "consider_traffic": False,
            "consider_time_windows": False
        }
        
        # Sample successful OptimizationResult DTO (as would be returned by the service)
        self.mock_successful_result_dto = OptimizationResult(
            status='success',
            total_distance=123.45,
            total_cost=67.89,
            routes=[], # Simplified, actual would have data
            detailed_routes=[ # This is what RouteOptimizationResponseSerializer.routes expects
                {
                    "vehicle_id": "vehicle1",
                    "total_distance": 123.45,
                    "total_time": 60.0,
                    "stops": ["depot", "customer1", "depot"],
                    "segments": [], # Simplified
                    "capacity_utilization": 0.1,
                    "estimated_arrival_times": {"customer1": 30},
                    "detailed_path": [[34.0522, -118.2437], [34.0523, -118.2438]]
                }
            ],
            unassigned_deliveries=[],
            statistics={"some_stat": "some_value"}
        )

    @patch('route_optimizer.api.views.OptimizationService.optimize_routes')
    def test_optimize_routes_success(self, mock_optimize_routes):
        mock_optimize_routes.return_value = self.mock_successful_result_dto

        response = self.client.post(self.optimize_url, self.valid_request_data, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'success')
        self.assertEqual(response.data['total_distance'], 123.45)
        # Check if 'routes' in response data corresponds to 'detailed_routes' from DTO
        self.assertEqual(len(response.data['routes']), 1)
        self.assertEqual(response.data['routes'][0]['vehicle_id'], "vehicle1")
        mock_optimize_routes.assert_called_once()
        # Further assertions on call arguments if needed

    @patch('route_optimizer.api.views.OptimizationService.optimize_routes')
    def test_optimize_routes_with_traffic_location_pairs(self, mock_optimize_routes):
        mock_optimize_routes.return_value = self.mock_successful_result_dto
        
        request_data_with_traffic = {
            **self.valid_request_data,
            "consider_traffic": True,
            "traffic_data": {
                "location_pairs": [["depot", "customer1"]],
                "factors": [1.5]
            }
        }
        response = self.client.post(self.optimize_url, request_data_with_traffic, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        args, kwargs = mock_optimize_routes.call_args
        # Location IDs: depot=0, customer1=1
        expected_traffic_service_data = {(0, 1): 1.5}
        self.assertEqual(kwargs['traffic_data'], expected_traffic_service_data)

    @patch('route_optimizer.api.views.OptimizationService.optimize_routes')
    def test_optimize_routes_with_traffic_segments(self, mock_optimize_routes):
        mock_optimize_routes.return_value = self.mock_successful_result_dto
        
        request_data_with_traffic = {
            **self.valid_request_data,
            "consider_traffic": True,
            "traffic_data": {
                "segments": {"depot-customer1": 2.0}
            }
        }
        response = self.client.post(self.optimize_url, request_data_with_traffic, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        args, kwargs = mock_optimize_routes.call_args
        # Location IDs: depot=0, customer1=1
        expected_traffic_service_data = {(0, 1): 2.0}
        self.assertEqual(kwargs['traffic_data'], expected_traffic_service_data)

    def test_optimize_routes_invalid_input(self):
        invalid_data = {"locations": [], "vehicles": [], "deliveries": []} 
        response = self.client.post(self.optimize_url, invalid_data, format='json')
        
        # Check that the view returned HTTP 400
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        
        # Check the content of the response data
        # The service should have returned an OptimizationResult DTO with status='error'
        self.assertEqual(response.data['status'], 'error')
        self.assertIn('statistics', response.data)
        self.assertIn('error', response.data['statistics'])
        # Check for the specific error message from the OptimizationService
        self.assertIn("Optimization failed: No locations provided", response.data['statistics']['error'])

    @patch('route_optimizer.api.views.OptimizationService.optimize_routes')
    def test_optimize_routes_service_exception(self, mock_optimize_routes):
        mock_optimize_routes.side_effect = Exception("Service exploded")

        response = self.client.post(self.optimize_url, self.valid_request_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn('error', response.data)
        # Update to the new generic error message
        self.assertEqual(response.data['error'], "An unexpected error occurred during route optimization. Please try again later.")

    @patch('route_optimizer.api.views.OptimizationService.optimize_routes')
    def test_optimize_routes_response_serializer_invalid(self, mock_optimize_routes):
        # Mock service to return a DTO that will make response serializer invalid
        # e.g. 'detailed_routes' is not a list of dicts
        faulty_dto = OptimizationResult(
            status='success',
            total_distance=100,
            total_cost=10,
            detailed_routes="not_a_list_of_route_dicts", # This will cause serializer to fail
            unassigned_deliveries=[],
            statistics={}
        )
        mock_optimize_routes.return_value = faulty_dto
        response = self.client.post(self.optimize_url, self.valid_request_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn('routes', response.data) # errors key should be 'routes' as per RouteOptimizationResponseSerializer


class RerouteViewTests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.reroute_url = reverse('reroute_vehicles_update') # Matches operation_id in RerouteView

        self.location_data1 = {"id": "depot", "name": "Main Depot", "latitude": 34.0522, "longitude": -118.2437, "is_depot": True}
        self.location_data2 = {"id": "customer1", "name": "Client One", "latitude": 34.0523, "longitude": -118.2438}
        self.location_data3 = {"id": "customer2", "name": "Client Two", "latitude": 34.0524, "longitude": -118.2439}
        
        self.vehicle_data1 = {"id": "vehicle1", "capacity": 100.0, "start_location_id": "depot"}
        
        self.delivery_data1 = {"id": "delivery1", "location_id": "customer1", "demand": 10.0}
        self.delivery_data2 = {"id": "delivery2", "location_id": "customer2", "demand": 15.0}

        self.current_routes_dict = { # This is what OptimizationResult.from_dict expects
            "status": "success",
            "total_distance": 200.0,
            "total_cost": 50.0,
            "routes": [["depot", "customer1", "customer2", "depot"]],
            "detailed_routes": [{
                "vehicle_id": "vehicle1", "stops": ["depot", "customer1", "customer2", "depot"], 
                "total_distance": 200.0, "total_time": 120.0, "segments": [],
                "capacity_utilization": 0.25, "estimated_arrival_times": {}
            }],
            "assigned_vehicles": {"vehicle1": 0},
            "unassigned_deliveries": [],
            "statistics": {"initial_stat": "value"}
        }

        self.base_reroute_request_data = {
            "locations": [self.location_data1, self.location_data2, self.location_data3],
            "vehicles": [self.vehicle_data1],
            "original_deliveries": [self.delivery_data1, self.delivery_data2],
            "current_routes": self.current_routes_dict
        }

        self.mock_successful_reroute_dto = OptimizationResult(
            status='success',
            total_distance=180.0, # Rerouted distance
            total_cost=45.0,
            detailed_routes=[{
                "vehicle_id": "vehicle1", "stops": ["depot", "customer2", "customer1", "depot"], 
                "total_distance": 180.0, "total_time": 110.0, "segments": [],
                "capacity_utilization": 0.25, "estimated_arrival_times": {},
                "detailed_path": []
            }],
            unassigned_deliveries=[],
            statistics={"rerouting_info": {"reason": "traffic"}}
        )

    @patch('route_optimizer.api.views.ReroutingService.reroute_for_traffic')
    def test_reroute_traffic_success(self, mock_reroute_for_traffic):
        mock_reroute_for_traffic.return_value = self.mock_successful_reroute_dto
        
        request_data = {
            **self.base_reroute_request_data,
            "reroute_type": "traffic",
            "traffic_data": { # Example traffic data, matches TrafficDataSerializer
                "segments": {"customer1-customer2": 1.8}
            }
        }
        response = self.client.post(self.reroute_url, request_data, format='json')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'success')
        self.assertEqual(response.data['total_distance'], 180.0)
        self.assertIn('rerouting_info', response.data['statistics'])
        
        args, kwargs = mock_reroute_for_traffic.call_args
        # Location IDs: depot=0, customer1=1, customer2=2
        # traffic_data_for_service expects index-based keys
        expected_traffic_data_service = {(1, 2): 1.8}
        self.assertEqual(kwargs['traffic_data'], expected_traffic_data_service)
        self.assertIsInstance(kwargs['current_routes'], OptimizationResult)


    @patch('route_optimizer.api.views.ReroutingService.reroute_for_delay')
    def test_reroute_delay_success(self, mock_reroute_for_delay):
        mock_reroute_for_delay.return_value = self.mock_successful_reroute_dto
        request_data = {
            **self.base_reroute_request_data,
            "reroute_type": "delay",
            "delayed_location_ids": ["customer1"],
            "delay_minutes": {"customer1": 30}
        }
        response = self.client.post(self.reroute_url, request_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        mock_reroute_for_delay.assert_called_once()
        args, kwargs = mock_reroute_for_delay.call_args
        self.assertEqual(kwargs['delayed_location_ids'], ["customer1"])
        self.assertEqual(kwargs['delay_minutes'], {"customer1": 30})

    @patch('route_optimizer.api.views.ReroutingService.reroute_for_roadblock')
    def test_reroute_roadblock_success(self, mock_reroute_for_roadblock):
        mock_reroute_for_roadblock.return_value = self.mock_successful_reroute_dto
        request_data = {
            **self.base_reroute_request_data,
            "reroute_type": "roadblock",
            "blocked_segments": [["customer1", "customer2"]]
        }
        response = self.client.post(self.reroute_url, request_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        mock_reroute_for_roadblock.assert_called_once()
        args, kwargs = mock_reroute_for_roadblock.call_args
        self.assertEqual(kwargs['blocked_segments'], [("customer1", "customer2")]) # Expects list of tuples

    def test_reroute_invalid_input(self):
        invalid_data = {**self.base_reroute_request_data}
        del invalid_data['current_routes'] # Make it invalid
        response = self.client.post(self.reroute_url, invalid_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    @patch('route_optimizer.api.views.ReroutingService.reroute_for_traffic')
    def test_reroute_service_exception(self, mock_reroute_for_traffic):
        mock_reroute_for_traffic.side_effect = Exception("Rerouting service crashed")
        request_data = {**self.base_reroute_request_data, "reroute_type": "traffic"}
        response = self.client.post(self.reroute_url, request_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn('error', response.data) # Ensure 'error' key is present
        # Update to the new generic error message
        self.assertEqual(response.data['error'], "An unexpected error occurred during rerouting. Please try again later.")

    @patch('route_optimizer.api.views.ReroutingService.reroute_for_traffic')
    def test_reroute_service_returns_none(self, mock_reroute_for_traffic):
        mock_reroute_for_traffic.return_value = None # Service returns None
        request_data = {**self.base_reroute_request_data, "reroute_type": "traffic"}
        response = self.client.post(self.reroute_url, request_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Invalid reroute type or no result obtained", response.data['error'])


class HealthCheckViewTests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.health_url = reverse('health_check_get') # Matches operation_id for health_check

    def test_health_check(self):
        response = self.client.get(self.health_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data, {"status": "healthy"})