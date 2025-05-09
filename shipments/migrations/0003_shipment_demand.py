# Generated by Django 5.2 on 2025-05-08 11:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('shipments', '0002_remove_shipment_destination_warehouse_id_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='shipment',
            name='demand',
            field=models.PositiveIntegerField(default=0, help_text='Amount of load required for this shipment (e.g., in kg or units)'),
        ),
    ]
