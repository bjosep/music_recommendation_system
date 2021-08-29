from utils.utils import get_img_and_track_url

def add_preview_url(popular_songs):
    table_html = get_table_html()
    popular_songs['Preview'] = popular_songs['id'].apply(lambda id: fill_html(*get_img_and_track_url(id)))
    popular_songs.drop(columns=['id'], inplace=True)
    popular_songs_html = table_html.format(table=popular_songs.to_html(classes='mystyle', escape=False))
    return popular_songs_html


def fill_html(img_url, track_url):
    html_code = f'''<figure><img src='{img_url}'  width="140" height="120"/><figcaption>\
    <a href="{track_url}" target="_blank" >{get_button_code()}</a></figcaption></figure>'''
    return html_code

def get_table_html():
    table_html = '''<html> <head><title>HTML Pandas Dataframe with CSS</title></head>
  <link rel="stylesheet" type="text/css">
  <body>{table}</body></html>.'''
    return table_html

def get_footer():
    footer = """
     <style>
    footer {
	visibility: hidden;
	}
    footer:after {
	content:'Contact: belyazyf@mail.uc.edu'; 
	visibility: visible;
	display: block;
	position: relative;
	#background-color: red;
	padding: 5px;
	top: 2px;
    }
    </style>
    """
    return footer


def get_button_code():
    code = '''<div style="text-align:center;"><svg version="1.1" id="play" \
    xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" \
    y="0px" height="40px" width="40px"viewBox="0 0 100 100" enable-background="new 0 0 100 100" xml:space="preserve">
    <path class="stroke-solid" fill="none" stroke="#ddbe72"  \
    d="M49.9,2.5C23.6,2.8,2.1,24.4,2.5,50.4C2.9,76.5,24.7,98,50.3,97.5c26.4-0.6,47.4-21.8,47.2-47.7\
    C97.3,23.7,75.7,2.3,49.9,2.5"/><path class="icon" fill="#ddbe72" d="M38,69c-1,0.5-1.8,0-1.8-1.1V32.1c0-1.1,0.8-1.6,\
    1.8-1.1l34,18c1,0.5,1,1.4,0,1.9L38,69z"/></svg></div>'''
    return code
