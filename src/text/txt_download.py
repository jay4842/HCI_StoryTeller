import urllib.request
import requests
from bs4 import BeautifulSoup, SoupStrainer
import os

# download with progress bar
def download_with_progress(link, save_path):
    with open(save_path, 'wb') as f:
        print('Downloading {}'.format(save_path))
        response = requests.get(link, stream=True)
        total_length = response.headers.get('content-length')
        size_KB = int(total_length) / 1024.0 # in KB (Kilo Bytes)
        size_MB = size_KB / 1024.0 # size in MB (Mega Bytes)
        if(total_length is None):
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)

            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = (50 * dl // total_length)
                print('\r[{}{}] {:.2f}Mb/{:.2f}Mb'.format('=' * done, ' ' * (50 - done), (dl/1024.0)/1024.0,size_MB), end='')
        f.close()
# Find all the links on the link in resp, and download them using download_with_progress
def download_poe(save='text_dataset/poe/'):
    page_url = 'http://www.textfiles.com/etext/AUTHORS/POE/'
    os.makedirs(save,exist_ok=True) # make our folder
    resp = urllib.request.urlopen(page_url)
    soup = BeautifulSoup(resp, from_encoding=resp.info().get_param('charset'))
    for link in soup.find_all('a', href=True):
        url = link['href']
        download_with_progress(link=page_url+url,save_path=save+(url.replace('.poe','.txt')))
