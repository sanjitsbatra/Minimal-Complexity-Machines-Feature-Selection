from datetime import datetime

def sterilize(input):
    input = input.replace("'","\\'")
    input = input.replace("\"","\\\"")
    return input
    
def dateToString():
    x = datetime.now()
    return str(x.year) + '-' + str(x.month).rjust(2,'0') + '-' + str(x.day).rjust(2,'0')
    
#Things we need:
#   name
#   refname
#   reflink
#   downloadlink
#   documentlink
#   description
#   date (we'll get this ourselves)
name = sterilize(str(raw_input("Name: ")))
refname = sterilize(str(raw_input("Reference Name: ")))
reflink = sterilize(str(raw_input("Reference Link: ")))
downloadlink = sterilize(str(raw_input("Download Link: ")))
doclink = sterilize(str(raw_input("Documentation Link: ")))
description = sterilize(str(raw_input("Description: ")))
date = dateToString()

sql = "INSERT INTO algorithms (name, reference, reference_link, download_link, documentation_link, description, time) Values ('" + name + "', '" + refname + "', '" + reflink + "', '" + downloadlink + "', '" + doclink + "', '" + description + "', '" + date + "');"

print "\n\n" + sql
#dateToString()
