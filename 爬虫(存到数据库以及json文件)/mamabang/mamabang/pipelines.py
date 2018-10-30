# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import json
import pymysql
import logging as log
class MamabangPipeline(object):
    # def __init__(self):
    #     self.filename=open('mmb.json','wb+')
    # def process_item(self, item, spider):
    #     text=json.dumps(dict(item),ensure_ascii=False)+',\n'
    #     self.filename.write(text.encode('utf-8'))
    #     return item
    # def closefilename(self):
    #     self.filename.close()
    def __init__(self):
        self.connent=pymysql.connect(
            host='localhost',
            port=3306,
            db='腾讯招聘',
            user='root',
            passwd='root',
            charset='utf8',
            use_unicode=True
        )
        #增删改查的方法
        self.cursor=self.connent.cursor()





    def process_item(self, item, spider):
        try:
            #开始去重
            self.cursor.execute(
                """select * from mamabang where email=%s""",
                item['email']
            )
            repetition=self.cursor.fetchone()
            if repetition:
                pass
            else:
                self.cursor.execute(
                    """insert into mamabang(name,email,author,num,time,content) value (%s,%s,%s,%s,%s,%s)""",
                    (item['name'],
                     item['email'],
                     item['author'],
                     item['num'],
                     item['time'],
                     item['content']
                     )
                )
                #提交命令
                self.connent.commit()
        except Exception as error:
            log(error)

        return item
