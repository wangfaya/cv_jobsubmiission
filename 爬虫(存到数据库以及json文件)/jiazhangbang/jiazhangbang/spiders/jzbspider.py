# -*- coding: utf-8 -*-
import scrapy
from ..items import JiazhangbangItem

class JzbspiderSpider(scrapy.Spider):
    name = 'jzbspider'
    allowed_domains = ['jzb.com']
    num=1
    url="http://www.jzb.com/bbs/forum-809-"

    start_urls = [url+str(num)+".html"]

    def parse(self, response):
        for each in response.xpath("//div[@class='bm_c']/form/table/tbody"):
            item=JiazhangbangItem()
            n=each.xpath("./tr/th/span/a/text()").extract()
            item['name']=n[0] if n else None
            e=each.xpath("./tr/th/span/a/@href").extract()
            item['email']=e[0] if e else None
            a=each.xpath("./tr/td[@class='by']/cite/a/text()").extract()
            item['author']=a[0] if a else None
            p=each.xpath("./tr/td[@class='by']/em/span/text()").extract()
            item['postedtime']=p[0] if p else None
            r=each.xpath("./tr/td[@class='num']/a/text()").extract()
            item['repliesnum']=r[0] if r else None
            all=each.xpath("./tr/td[@class='num']/em/text()").extract()
            item['allnum']=all[0] if all else None
            lp = each.xpath("./tr/td[@class='by']/cite/a/text()").extract()
            item['lastpost']=lp[0] if lp else None
            lt = each.xpath("./tr/td[@class='by']/em/a/text()").extract()
            item['lasttime']=lt[0] if lt else None
            nexurl=response.urljoin(item['email'])
            yield scrapy.Request(str(nexurl),callback=self.contents,meta={'item':item})
            # yield item
        if self.num<671:
            self.num+=1
        yield scrapy.Request(self.url+str(self.num)+".html",callback=self.parse)
    def contents(self,response):
        item=response.meta['item']
        str1 = ''
        for i in range(len(response.xpath("//td[@class='t_f']/text()"))):

            str1+=response.xpath("//td[@class='t_f']/text()").extract()[i].strip()
        item['content']=str1 if str1 else None
        # item['content']=response.xpath("//td[@class='t_f']/text()").extract()[0]
        yield item




