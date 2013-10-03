#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = u'Paul Ivanov'
SITENAME = u'pyarbus'
SITESUBTITLE = u'eyetracking analysis software written in python.'
SITEURL = ''

TIMEZONE = 'America/Los_Angeles'

DEFAULT_LANG = u'en'

GITHUB_URL = 'http://github.com/ivanov/pyarbus'
THEME = './themes/notmyidea'



# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None

# Blogroll
LINKS =  (('development', 'https://github.com/ivanov/pyarbus'),
          ('buildbot status', 'http://nipy.bic.berkeley.edu/waterfall?show=pyarbus-py2.6'),
          ('SciPy Stack', 'http://scipy.org/about.html'),
          ('Python.org', 'http://python.org/'),
          )

# Social widget
#SOCIAL = (('You can add links in your config file', '#'),
#          ('Another social link', '#'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True
