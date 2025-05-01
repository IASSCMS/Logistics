from django.test import TestCase
from route_optimizer.services.path_annotation_service import PathAnnotator

class DummyPathFinder:
    def calculate_shortest_path(self, graph, from_node, to_node):
        return [from_node, to_node], 5

class PathAnnotatorTest(TestCase):
    def test_annotate(self):
        graph = {'A': {'B': 5}, 'B': {'A': 5}}
        result = {'routes': [['A', 'B']]}
        
        annotator = PathAnnotator(DummyPathFinder())
        annotator.annotate(result, graph)
        
        self.assertIn('detailed_routes', result)
        self.assertEqual(len(result['detailed_routes']), 1)
        self.assertEqual(result['detailed_routes'][0]['segments'][0]['distance'], 5)
