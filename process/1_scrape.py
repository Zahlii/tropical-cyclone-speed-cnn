import time
import socket
import requests.packages.urllib3.util.connection as urllib3_cn
import numpy as np
import time
from functools import wraps
import cv2
import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import List
from settings import DATA_DIR, IMAGE_SIZE


def retry(exceptions, tries=4, delay=3, backoff=2, logger=None):
    """
    Retry calling the decorated function using an exponential backoff.

    Args:
        exceptions: The exception to check. may be a tuple of
            exceptions to check.
        tries: Number of times to try (not retry) before giving up.
        delay: Initial delay between retries in seconds.
        backoff: Backoff multiplier (e.g. value of 2 will double the delay
            each retry).
        logger: Logger to use. If None, print.
    """
    def deco_retry(f):

        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    msg = '{}, Retrying in {} seconds...'.format(e, mdelay)
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return deco_retry


def allowed_gai_family():
    """
     https://github.com/shazow/urllib3/blob/master/urllib3/util/connection.py
    """
    family = socket.AF_INET
    # if urllib3_cn.HAS_IPV6:
    #    family = socket.AF_INET6 # force ipv6 only if it is available
    return family


urllib3_cn.allowed_gai_family = allowed_gai_family

s = requests.Session()


@retry((requests.exceptions.HTTPError, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout),3,2.5,2)
def get_bytes(url):
    #url = url.replace("https:","http:")
    t = time.time()

    res = s.get(url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept-Language': 'de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7,sv;q=0.6',
                    'Connection': 'keep-alive'
                }, timeout=2.5)

    if 'jpg' not in url:
        print('HTTPS |', url, '|', time.time() - t)
    return res.content


def get_html(url):
    h = get_bytes(url)
    return BeautifulSoup(h, 'html.parser')


def read_and_crop(file):

    img = cv2.imread(file)
    if img is None:
        print('Deleting', file)
        os.remove(file)
        return None
    # original img: 1024 x 1024 > new img: central 800 x 800
    img = img[112:912, 112:912]
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    nnz = np.count_nonzero(img)
    perc_black = 1 - nnz/img.size

    if perc_black >= 0.35:
        print('Deleting due to too many black pixels',perc_black*100, file)
        with open(file,'w') as f:
            f.write('')
        return

    cv2.imwrite(file, img)
    return img


# @retry(urllib.request.HTTPError,3,2,2)
def save(url, path):
    # url = url.replace("https:", "http:")
    path = os.path.join(DATA_DIR, path)
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

    if os.path.exists(path):
        return

    # print('Saving',os.path.basename(path))

    try:
        b = get_bytes(url)
        with open(path, 'wb') as f:
            f.write(b)

        read_and_crop(path)
    except Exception as e:
        return


def filter_image_links(url):
    image_links = get_outbound_links(url)
    urls = []

    for h in image_links:
        if not h.endswith(".jpg"):
            continue

        if 'INVEST' in h or 'NONAME' in h:
            continue

        if 'kts' not in h:
            continue

        if 'NAmb' in h:
            continue

        #if 'pc.' not in h:
        #    continue

        #pc = int(h.split('pc.')[0].split('.')[-1])
        #if pc < 80:
        #    continue

        full_url = url + h
        # https://www.nrlmry.navy.mil/htdocs_dyn_apache/PUBLIC/tc_pages/thumbs/medium/tc18/EPAC/02E.ALETTA/vapor/geo/1km/20180609.1630.goes15.x.wv1km.02EALETTA.90kts-970mb-163N-1129W.93pc.jpg

        urls.append((full_url, h))

    return urls


def fetch_storm(year, basin, name):
    year = str(year)
    y = year[2:4]

    url_base = 'https://www.nrlmry.navy.mil/tcdat/tc' + y + '/' + basin + '/' + name + '/'

    url_wv = url_base + 'vapor/geo/1km/'
    url_ir = url_base + 'ir/geo/1km/'

    wv_urls = filter_image_links(url_wv)
    ir_urls = filter_image_links(url_ir)

    l = len(wv_urls)

    if l < 1:
        return

    with ThreadPoolExecutor(max_workers=16) as tpe:
        urls = [u_full for u_full, u_frag in wv_urls]
        paths = [year + '/' + basin + '/' + name + '/wv/' + u_frag for u_full, u_frag in wv_urls]
        res = list(tqdm(tpe.map(save, urls, paths), desc='Downloading WV', total=len(urls)))

    with ThreadPoolExecutor(max_workers=16) as tpe:
        urls = [u_full for u_full, u_frag in ir_urls]
        paths = [year + '/' + basin + '/' + name + '/ir/' + u_frag for u_full, u_frag in ir_urls]
        res = list(tqdm(tpe.map(save, urls, paths), desc='Downloading IR', total=len(urls)))


def get_outbound_links(url):
    page = get_html(url)
    links = page.find_all('a')
    links: List[str] = [l['href'] for l in links]
    return [l for l in links if not l.startswith('?') and 'mailto' not in l and not l.startswith('/') and 'tar.gz' not in l]


def fetch_season(year):
    print('Season', year)

    overview = 'https://www.nrlmry.navy.mil/tcdat/tc%02d/' % (year - 2000)

    basins = get_outbound_links(overview)
    tb = len(basins)

    for ib, basin in enumerate(basins):

        storms = get_outbound_links(overview + basin)

        basin = basin.replace('/', '')
        print('\tBasin', basin, ib + 1, '/', tb)

        ts = len(storms)

        for ist, storm in enumerate(storms):

            storm = storm.replace('/', '')
            print('\t\tStorm', storm, ist + 1, '/', ts)

            fetch_storm(year, basin, storm)


for i in range(2016, 2019, 1):
    fetch_season(i)
print('Done')