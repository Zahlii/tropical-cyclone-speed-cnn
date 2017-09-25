import os
import urllib.request
from bs4 import BeautifulSoup
import time
import urllib.parse as urlparse
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor

data_dir = os.path.abspath('./NOAA/')





ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36'


def get_html(url):
    t = time.time()
    req = urllib.request.Request(
        url,
        data=None,
        headers={
            'User-Agent': ua
        }
    )
    with urllib.request.urlopen(req,context=ctx) as c:
        h = c.read()

    print('HTTP |', url, '|',time.time()-t)
    return BeautifulSoup(h,'html.parser')


def save(url, path):
    path = os.path.join(data_dir,path)
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)

    if os.path.exists(path):
        return

    print('Saving',os.path.basename(path))

    try:
        return urllib.request.urlretrieve(url, path)
    except Exception as e:
        print(e)
        return

def save_index_page(url):
    html = get_html(url)
    a = html.find_all('a')
    urls = []
    for x in a:
        h = x['href']
        if not h.endswith(".jpg"):
            continue

        if 'INVEST' in h or 'NONAME' in h:
            continue

        if not 'kts' in h:
            continue

        full_url = url + h
        urls.append((full_url,h))

    return urls


def fetch_storm(year, basin, name):
    print('\t',year, basin, name)

    y = year[2:4]

    url_base = 'https://www.nrlmry.navy.mil/tcdat/tc'+y+'/'+basin+'/'+name+'/'
    url_wv = url_base + 'vapor/geo/1km/'
    url_ir = url_base + 'ir/geo/1km/'

    wv_urls = save_index_page(url_wv)
    ir_urls = save_index_page(url_ir)

    l = len(wv_urls)
    i = 1
    for u_full, u_frag in wv_urls:
        if i % 50 == 0:
            print('\t\tWV',i,'/',l)
        save(u_full, year + '/' + basin + '/' + name + '/wv/' + u_frag)
        i += 1

    l = len(ir_urls)
    i = 1
    for u_full, u_frag in ir_urls:
        if i % 50 == 0:
            print('\t\tIR',i,'/',l)
        save(u_full, year + '/' + basin + '/' + name + '/ir/' + u_frag)
        i += 1


def fetch_season(year):
    print('Season',year)

    overview = 'https://www.nrlmry.navy.mil/tc-bin/tc_list_storms2.cgi?ARCHIVE=all&AGE=Prev&YEAR=' + str(year) + '&TYPE=ssmi&SIZE=FULL'

    res = get_html(overview)
    td = res.find_all('td')
    ltd = len(td)

    with ThreadPoolExecutor(max_workers=32) as tpe:
        for _,l in enumerate(td):
            for a in l.find_all('a'):
                print('\tStarting Storm',_,'/',ltd)
                h = a['href']
                p = urlparse.urlparse(h)
                p = urlparse.parse_qs(p.query)
                if 'BASIN' not in p:
                    continue
                tpe.submit(fetch_storm,str(year),p['BASIN'][0],p['STORM_NAME'][0])


for i in range(2014,2018,1):
    fetch_season(i)

print('Done')