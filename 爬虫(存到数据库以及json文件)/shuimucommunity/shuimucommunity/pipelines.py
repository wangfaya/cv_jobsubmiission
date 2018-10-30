# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import pymysql
import json
import logging as log
class ShuimucommunityPipeline(object):
    def __init__(self):
        self.con=pymysql.connect(
            host='localhost',
            port=3306,
            db='腾讯招聘',
            user='root',
            passwd='root',
            use_unicode=True
        )
        #通过cursor方法进行增删改查
        self.cursor=self.con.cursor()




    def process_item(self, item, spider):
        try:
            self.cursor.execute(
                """select * from shuimu where email = %s""",item['email']
            )
            res=self.cursor.fetchone()
            if res:
                pass
            else:
                self.cursor.execute(
                    """insert into shuimu(name,email,postedtime,author,replynum,replytime,replyauthor,content) value (%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (item['name'],
                     item['email'],
                     item['postedtime'],
                     item['author'],
                     item['replynum'],
                     item['replytime'],
                     item['replyauthor'],
                     item['content']

                     )
                )
                self.con.commit()

        except Exception as error:
            log(error)
        return item

    # def __init__(self):
    #     self.filename=open('shuimuall.json','wb+')
    # def process_item(self,item,spider):
    #     text=json.dumps(dict(item),ensure_ascii=False)+',\n'
    #     self.filename.write(text.encode('utf-8'))
    #
    #     return item
    # def closefilename(self):
    #     self.filename.close()

