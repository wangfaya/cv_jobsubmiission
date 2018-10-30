# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import json
import pymysql
import logging as log
class JiazhangbangPipeline(object):

    # def __init__(self):
    #     self.filename=open("jzball.json",'wb+')
    #
    # def process_item(self, item, spider):
    #     text=json.dumps(dict(item),ensure_ascii=False)+",\n"
    #     self.filename.write(text.encode("utf-8"))
    #     return item
    # def closespider(self,spider):
    #     self.filename.close()
    def __init__(self):
        self.connect=pymysql.connect(
            host='localhost',
            port=3306,
            db='腾讯招聘',
            user='root',
            passwd='root',
            charset='utf8',
            use_unicode=True
        )
        #通过cursor执行增删改查
        self.cursor=self.connect.cursor()
    def process_item(self,item,spider):
        try:
            #查重处理
            self.cursor.execute(
                """select * from parentshelp where email=%s""",
                item['email'])
            repetition=self.cursor.fetchone()
            if repetition:
                pass
            else:
                self.cursor.execute(
                    """insert into parentshelp(name,email,author,postedtime,repliesnum,allnum,lastpost,lasttime,content) value (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (item['name'],
                     item['email'],
                     item['author'],
                     item['postedtime'],
                     item['repliesnum'],
                     item['allnum'],
                     item['lastpost'],
                     item['lasttime'],
                     item['content']

                     )
                )
                #提交sql语句
                self.connect.commit()
        except Exception as error:
            log(error)
        return item
