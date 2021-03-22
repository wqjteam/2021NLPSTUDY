s="a good       example"
a = ""
flag = True
for i in range(0, len(s)):
    if (s[i] == ' '):
        if flag == True:
            continue
        else:
            flag = True
            a = a + s[i]
    else:
        flag = False
        a = a + s[i]
print(a)