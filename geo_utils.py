import requests
import json


def get_country(place_name, api_key, cache_json=None):
    if cache_json:
        try:
            cache = json.load(open(cache_json))
        except Exception:
            cache = dict()
            json.dump(cache, open(cache_json, 'w'))

    if cache_json and place_name in cache:
        return cache[place_name]

    place_api_url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"

    params = {
        'input': place_name,
        'inputtype': 'textquery',

        'fields': 'formatted_address',
        'key': api_key
    }

    response = requests.get(place_api_url, params=params)
    response_json = response.json()


    if response_json.get('status') == 'OK':

        address_components = response_json['candidates'][0].get(
            'formatted_address').split(', ')

        if address_components[-1].isdigit():
            country = address_components[-2]
        else:
            country = address_components[-1]

        if cache_json:
            cache[place_name] = country
            with open(cache_json, 'w') as f:
                f.write(json.dumps(cache, indent=4))
        return country

    else:
        return None


def get_location_info(place_name, api_key):


    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={place_name}&key={api_key}"

    try:
        response = requests.get(geocode_url)
        data = response.json()


        if data['status'] == 'OK':

            location = data['results'][0]['geometry']['location']
            latitude = location['lat']
            longitude = location['lng']


            country_code = None
            for component in data['results'][0]['address_components']:
                if 'country' in component['types']:
                    country_code = component['short_name']
                    break

            return latitude, longitude, country_code
        else:
            print(f"Error: {data['status']}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
