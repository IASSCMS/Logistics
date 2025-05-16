import json
import logging
import uuid

from confluent_kafka import Consumer, KafkaException
from django.conf import settings
from shipments.models import Shipment


def create_kafka_consumer():
    return Consumer({
        'bootstrap.servers': settings.KAFKA_BROKER_URL,
        'group.id': 'shipment_consumer_group',
        'auto.offset.reset': 'earliest',
    })


def handle_order_created(event):
    order_id = event.get("order_id")
    origin = event.get("origin")  # Expecting {"lat": ..., "lng": ...}
    destination = event.get("destination")
    demand = event.get("demand", 0)

    # Basic validation
    if not (order_id and origin and destination):
        logging.error("Invalid order event payload: missing fields")
        return

    if not all(k in origin for k in ("lat", "lng")) or not all(k in destination for k in ("lat", "lng")):
        logging.error("Origin/destination must include lat/lng")
        return

    if not isinstance(demand, int) or demand < 0:
        logging.warning(f"Invalid or missing demand for order {order_id}. Defaulting to 0.")
        demand = 0

    Shipment.objects.create(
        shipment_id=str(uuid.uuid4())[:12],
        order_id=str(order_id),
        origin=origin,
        destination=destination,
        demand=demand,
        status='pending'
    )
    logging.info(f"Shipment created for order {order_id} with demand {demand}")


def start_order_consumer():
    consumer = create_kafka_consumer()
    consumer.subscribe(['orders.created'])

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())

            event = json.loads(msg.value().decode('utf-8'))
            handle_order_created(event)
    except KeyboardInterrupt:
        print("Kafka consumer stopped")
    finally:
        consumer.close()


def run_consumer_once():
    consumer = create_kafka_consumer()
    consumer.subscribe(['orders.created'])
    msg = consumer.poll(timeout=5.0)
    if msg and not msg.error():
        event = json.loads(msg.value().decode('utf-8'))
        handle_order_created(event)
    consumer.close()
