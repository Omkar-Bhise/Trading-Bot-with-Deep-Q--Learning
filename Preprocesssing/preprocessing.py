import time
import urllib.request
import pandas as pd
# Build the cookie handler
cookier = urllib.request.HTTPCookieProcessor()
opener = urllib.request.build_opener(cookier)
urllib.request.install_opener(opener)

# Cookie and corresponding crumb
_cookie = None
_crumb = None

# Headers to fake a user agent
_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.101 Safari/537.36'
}


def _get_cookie_crumb():
    '''
    This function perform a query and extract the matching cookie and crumb.
    '''
    global cookier, _cookie, _crumb

    # Perform a Yahoo financial lookup on SP500
    cookier.cookiejar.clear()
    req = urllib.request.Request(
        'https://finance.yahoo.com/quote/^GSPC', headers=_headers)
    f = urllib.request.urlopen(req, timeout=5)
    alines = f.read().decode('utf-8')

    # Extract the crumb from the response
    cs = alines.find('CrumbStore')
    cr = alines.find('crumb', cs + 10)
    cl = alines.find(':', cr + 5)
    q1 = alines.find('"', cl + 1)
    q2 = alines.find('"', q1 + 1)
    crumb = alines[q1 + 1:q2]
    _crumb = crumb

    # Extract the cookie from cookiejar
    for c in cookier.cookiejar:
        if c.domain != '.yahoo.com':
            continue
        if c.name != 'B':
            continue
        _cookie = c.value

    # Print the cookie and crumb
    #print('Cookie:', _cookie)
    #print('Crumb:', _crumb)


def load_yahoo_quote(ticker, begindate, enddate, info='quote'):
    '''
    This function load the corresponding history/divident/split from Yahoo.
    The "begindate" and "enddate" are in the format of YYYYMMDD.
    The "info" can be "quote" for price, "divident" for divident events,
    or "split" for split events.
    '''
    # Check to make sure that the cookie and crumb has been loaded
    global _cookie, _crumb
    if _cookie == None or _crumb == None:
        _get_cookie_crumb()

    # Prepare the parameters and the URL
    tb = time.mktime((int(begindate[0:4]), int(
        begindate[4:6]), int(begindate[6:8]), 4, 0, 0, 0, 0, 0))
    te = time.mktime((int(enddate[0:4]), int(
        enddate[4:6]), int(enddate[6:8]), 18, 0, 0, 0, 0, 0))

    param = dict()
    param['period1'] = int(tb)
    param['period2'] = int(te)
    param['interval'] = '1d'
    if info == 'quote':
        param['events'] = 'history'
    elif info == 'dividend':
        param['events'] = 'div'
    elif info == 'split':
        param['events'] = 'split'
    param['crumb'] = _crumb
    params = urllib.parse.urlencode(param)
    url = 'https://query1.finance.yahoo.com/v7/finance/download/{}?{}'.format(
        ticker, params)
    # print(url)
    req = urllib.request.Request(url, headers=_headers)

    # Perform the query
    # There is no need to enter the cookie here, as it is automatically handled by opener
    f = urllib.request.urlopen(req, timeout=5)
    alines = f.read().decode('utf-8')
    # print(alines)
    return alines.split('\n')

def load_stock_data(stock_ticker):
	all_data=[]
	for data in load_yahoo_quote(stock_ticker, '20170722', '20180725')[1:]:
    		all_data.append(data.split(","))

	scraped_df=pd.DataFrame(all_data,columns=load_yahoo_quote(stock_ticker, '20170722', '20180725')[0].split(","))
	scraped_df=scraped_df.set_index('Date')
	scraped_df=scraped_df.astype('float32')
	return(scraped_df)