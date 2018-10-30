# -*- coding: utf-8 -*-
import scrapy
from ..items import MamabangItem

class MamaspiderSpider(scrapy.Spider):
    name = 'mamaspider'
    allowed_domains = ['mmbang.com']
    num=1
    start_urls = ['https://www.mmbang.com/bang/623/p'+str(num)+'#topics']

    def parse(self, response):
        for each in response.xpath("//div[@class='span-16 first last clear']/ul"):
            item=MamabangItem()
            n=each.xpath("./li[@class='topic_title']/a/text()").extract()
            item['name'] = n[0].strip() if n else None
            e=each.xpath("./li[@class='topic_title']/a/@href").extract()
            item['email']='https://www.mmbang.com'+e[0] if e else None
            a=each.xpath("./li[@class='topic_author']/text()").extract()
            item['author']=a[0].strip() if a else None
            nm=each.xpath("./li[@class='topic_posts number']/text()").extract()
            item['num']=nm[0].strip() if nm else None
            tm=each.xpath("./li[@class='topic_updated_time number']/text()").extract()
            item['time']=tm[0].strip() if tm else None
            newurl=response.urljoin(item['email'])
            yield scrapy.Request(newurl,callback=self.content,meta={'item':item})
        if self.num < 18:
            self.num +=1
        yield scrapy.Request('https://www.mmbang.com/bang/566/p'+str(self.num)+'#topics',callback=self.parse)
    def content(self,response):
        item=response.meta['item']

        con=response.xpath("//div[@class='art_answer']/text()").extract()

        str=''
        for co in con:
            str+=co
        str1=''
        for st in str.split():
            str1+=st
        item['content']=str1 if str1 else None
        yield item


