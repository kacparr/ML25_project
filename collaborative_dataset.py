import requests
import pandas as pd
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.getenv('LASTFM_API_KEY')
LASTFM_USERNAME = os.getenv('LASTFM_USERNAME')

# create collaborative filtering dataset of last fm user scrobbles 

def fetch_user_info(friendlist):
    data = []
    for i, friend in enumerate(friendlist):
        print(f'fetching info of user {friend} {i}/{len(friendlist)}')
        info_url = f"""
            http://ws.audioscrobbler.com/2.0/?method=user.getInfo
            &user={friend}
            &api_key={API_KEY}
            &format=json
        """
        try:
            info_response = requests.get(info_url)
            info_rawdata = info_response.json().get('user', {})
            user_data = {
                'name': info_rawdata.get('name', None),
                'realname': info_rawdata.get('realname', None),
                'country': info_rawdata.get('country', None), 
                'playcount': int(info_rawdata.get('playcount', 0)), 
                'artist_count': int(info_rawdata.get('artist_count', 0)), 
                'album_count': int(info_rawdata.get('album_count',0))
                }
            data.append(user_data)
            time.sleep(0.25)
        except requests.exceptions.RequestException as e:
            print(f"Request error for user {friend}: {e}")
        except Exception as e:
            print(f"Unknown error for user {friend}: {e}")
    df = pd.DataFrame(data)
    return df

def fetch_friends(user:str):
    friends = []
    for i in range(1,10):
        url = f"""
            http://ws.audioscrobbler.com/2.0/?method=user.getFriends
            &user={user}
            &page={i}
            &api_key={API_KEY}
            &format=json
        """
        response = requests.get(url)
        data = response.json()

        friend_list = [user.get("name") for user in data.get('friends', {}).get('user', {})]
        if not friend_list: break
        friends.extend(friend_list)
        time.sleep(0.25)
    return friends

def fetch_kouczi_network():
    network = []
    kouczi_friends = fetch_friends("kacparr")
    network.extend(kouczi_friends)
    for friend in kouczi_friends:
        print(f"fetching friends of {friend}:")
        second_degree_friends = fetch_friends(friend)
        for second_friend in second_degree_friends:
            if second_friend not in network:
                network.append(second_friend)
    return network

def clear_network(network):
    filtered = network.dropna(subset=['name'])
    filtered = filtered.query('playcount > 5000 and artist_count > 0 and playcount / artist_count < 2000')
    return filtered


def fetch_recents_page(user:str, page:int, limit:int=200):
    try:
        url=f"""
            http://ws.audioscrobbler.com/2.0/?method=user.getrecenttracks
            &user={user}
            &limit={limit}
            &page={page}
            &api_key={API_KEY}
            &format=json
        """
        response = requests.get(url=url)
        data = response.json()
        if 'error' in data:
            if data['error'] == 17:
                raise ValueError(f"{user}'s scrobbles set to private, skipping")
            raise ValueError(f"API error {data["error"]} for user {user} page{page}: {data.get('message')}")
        
        return data.get('recenttracks', {}).get('track', [])  
    except requests.exceptions.RequestException as e:
            message = f"Request error for user {user} page {page}: {str(e)}"
            print(message)
            with open("error_log.txt", "a") as f:
                f.write(f"{message}\n")

            return None
    except (KeyError, ValueError) as e:
        message = f"Data parsing error! ({str(e)})"
        print(message)
        with open("error_log.txt", "a") as f:
            f.write(f"{message}\n")

        return None
    except Exception as e:
        message = f"Unknown error for user {user} page {page}: {str(e)}"
        with open("error_log.txt", "a") as f:
            f.write(f"{message}\n")
        print(message)

        return None
    

def fetch_user_recents(user, limit=200, pages=25): #5000 scrobbles 
    user_recents = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        futures = {executor.submit(fetch_recents_page, user, page, limit): page for page in range(1, pages+1)}
  
        for future in as_completed(futures):
            tracks = future.result() # returned data.get('recenttracks', {}).get('track', []) or None
            page = futures[future]
            print(f"Fetching recent scrobbles for user {user}... {page/pages*100}% completed")
            if tracks:
                for track in tracks:
                    user_recents.append({
                        'user': user,
                        'artist': track['artist']['#text'],
                         'artist_mbid': track['artist']['mbid'],
                          'name': track['name'],
                         'name_mbid': track['mbid'],
                         'album': track['album']['#text'],
                         'album_mbid': track['album']['mbid'],
                         'date': int(track['date']['uts'] if 'date' in track else 0)
                       })
            else:
                  return user_recents
        time.sleep(0.25 / 5)
    return user_recents

def fetch_recent_dataset(network:pd.DataFrame):
    all_recents = []
    users_db = network["name"]

    with ThreadPoolExecutor(max_workers=5) as executor:
        user_futures = {}
        user_futures = {executor.submit(fetch_user_recents, user): user for user in users_db}

        for i, future in enumerate(as_completed(user_futures), 1):
              user = user_futures[future]
              user_recent = future.result()
              all_recents.extend(user_recent)
              print(f"{user} ({i}/{len(users_db)} - {len(user_recent)} scrobbles)")

    
    df = pd.DataFrame(all_recents)
    df.to_csv("lastfm_recent_tracks_test.csv", index=False)



# network = fetch_kouczi_network()
# fetch_user_info(network).to_csv('lastfm_fetched_users.csv')
# network = pd.read_csv("lastfm_fetched_users.csv", index_col=0)
# network = clear_network(network=network)
# fetch_recent_dataset(network=network)

recents = fetch_user_recents("kacparr",200,10)
recents = pd.DataFrame(recents)
recents.to_csv("lastfm_recent_tracks_user.csv", index=False) 
