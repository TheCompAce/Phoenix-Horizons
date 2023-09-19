import json

# Load the JSON data
with open('design.json') as f:
    data = json.load(f)

# Extract the main service list
main_service_list = [service['name'] for service in data['service']]

# Initialize an empty list to store services that are referenced but not found in the main service list
missing_services = []

# Check each service's dependencies and service consumers
for service in data['service']:
    dependencies = service.get('dependencies')
    if dependencies is not None:
        for dependency in dependencies:
            if dependency['name'] not in main_service_list and dependency['name'] not in missing_services:
                missing_services.append(dependency['name'])
    consumers = service.get('serviceConsumers')
    if consumers is not None:
        for consumer in consumers:
            if consumer['name'] not in main_service_list and consumer['name'] not in missing_services:
                missing_services.append(consumer['name'])

# Print the missing services
print('Services that are referenced but not found in the main service list:')
for service in missing_services:
    print(service)
