#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-

import pexpect


def main(server):
    command = 'ssh -p %s %s@%s' % (server['port'], server['username'], server['hostname'])
    process = pexpect.spawn(command, timeout=30)
    print(f'命令: {command}')
    expect_list = [
        'yes/no',
        'password:',
        pexpect.EOF,
        pexpect.TIMEOUT,
    ]
    index = process.expect(expect_list)
    print(f'匹配到: {index} => {expect_list[index]}')
    if index == 0:
        process.sendline("yes")
        expect_list = [
            'password:',
            pexpect.EOF,
            pexpect.TIMEOUT,
        ]
        index = process.expect(expect_list)
        print(f'匹配到: {index} => {expect_list[index]}')
        if index == 0:
            process.sendline(server['password'])
            process.interact()
        else:
            print('EOF or TIMEOUT')
    elif index == 1:
        process.sendline(server['password'])
        process.interact()
    else:
        print('EOF or TIMEOUT')


if __name__ == '__main__':
    server = {
        'hostname': '192.168.1.100',
        'port': '22',
        'username': 'admin',
        'password': 'ABuklhsfnVyxI',
    }
    main(server)
