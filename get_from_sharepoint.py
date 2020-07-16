import json
from O365 import Account
import os
from zipfile import ZipFile

with open('office365_credentials.json', 'r') as infile:
    data = json.load(infile)
    client_id = data['client_id']
    client_secret = data['client_secret']
    tenant_id = data['tenant_id']

acc = Account((client_id, client_secret))

if acc.authenticate(scopes=['basic', 'Files.ReadWrite.All']):
    print('Authenticated!')
    storage = acc.storage()
    drives = storage.get_drives()
    my_drive = storage.get_default_drive()
    root_folder = my_drive.get_root_folder()
    annotations_folder = my_drive.get_item_by_path('/MERC Data/Annotated Images/')

    if not os.path.isdir('./data-cache/'):
        os.mkdir('./data-cache/')
        if not os.path.isdir('./data-cache/raw/'):
            os.mkdir('./data-cache/raw/')

    for item in annotations_folder.get_items(limit=50):
        if item.is_file:
            print(f'Downloading {item.name}')
            item.download(to_path='./data-cache/')

            print(f'Extracting {item.name}')
            with ZipFile(f'./data-cache/{item.name}', 'r') as zip_ref:
                zip_ref.extractall('./data-cache/raw/')
                print(f'Done extracting {item.name}')
    pass


