# Field mapping and descriptions for LLM prompt construction

from dataclasses import field


FIELD_MAPPINGS = {
    "deliveries": {
        "delivery date": {"field": "actual_delivery_date", "desc": "The date the delivery actually happened."},
        "actual delivery date": {"field": "actual_delivery_date", "desc": "The date the delivery actually happened."},
        "promised delivery date": {"field": "promised_delivery_date", "desc": "The date the delivery was promised to the customer."},
        "order date": {"field": "order_date", "desc": "The date the order was placed."},
        "customer name": {"field": "customer_name", "desc": "The name of the customer."},
        "status": {"field": "status", "desc": "The current status of the delivery (e.g., Delivered, Failed)."},
        "amount": {"field": "amount", "desc": "The total amount for the order."},
        "delivery address": {"field": "delivery_address_line1", "desc": "The first line of the delivery address."},
        "delivery address 2": {"field": "delivery_address_line2", "desc": "The second line of the delivery address."},
        "city": {"field": "city", "desc": "The city for delivery."},
        "state": {"field": "state", "desc": "The state for delivery."},
        "pincode": {"field": "pincode", "desc": "The postal code for delivery."},
        "order id": {"field": "order_id", "desc": "The unique order identifier."},
        "customers": {"field": "customer_name", "desc": "Customers or customer name should be fetched from customer_name field."},
        "failed" : {"field"  : "failure_reason", "desc": "if there why the delivery failed, it is mentioned in failure_reason field."}
    },
    # Add mappings for other collections as needed
}
