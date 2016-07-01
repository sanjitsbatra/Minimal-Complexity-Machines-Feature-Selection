#!/usr/bin/env python

final = ""
repl = "<br/>&nbsp;&nbsp; "

x = str(raw_input(""))
while x != "":
    final = final + x
    x = str(raw_input(""))
    
final = final.replace(",", "," + repl, 1)
final = final.replace("},", "}," + repl)
final = final[:-2] + "}<br/>}"

print "\n\n\n" + final