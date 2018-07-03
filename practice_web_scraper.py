import urllib2
from bs4 import BeautifulSoup

#Save the name of the website
wiki = "https://en.wikipedia.org/wiki/List_of_state_and_union_territory_capitals_in_India"

#Open the website for mining
page = urllib2.urlopen(wiki)

#Parse the HTML
soup = BeautifulSoup(page)

#Display all the HTML
#print soup.prettify()

print soup.title()

