import itertools
#key = '0123456789qwertyuiopasdfghjklzxcvbnm'#密码包含这些字符
key = '0123456789'#密码包含这些字符
passwords = itertools.product(key,repeat = 8)
f = open('C:/Users/jsyzdlf/Desktop/password.txt', 'a')
for i in passwords:
    f.write("".join(i))
    f.write('\n')
f.close()