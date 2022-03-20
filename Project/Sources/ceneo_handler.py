"""
Back-end Class to handle the data collecting process
"""
import time
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

MAIN_PAGE_LINK = 'https://www.ceneo.pl'
SPEC_SUFFIX = "#tab=spec"
CENEO_NEXT_PAGE_STYLE = ";0020-30-0-0-"
PRODUCT_PER_PAGE = 30
COMPLETE_PROGRESS_BAR = 100


def request_page_data_soup(page_url):
    """
    Method that uses the library requests and BeautifulSoup to get data from page at given page_url.
    :param string page_url: given page url.

    :return BeautifulSoup: returns BeautifulSoup object with data from collected page.
    """
    page = requests.get(page_url)
    page_soup = BeautifulSoup(page.content, 'html.parser', multi_valued_attributes=None)
    return page_soup


class CeneoHandler:
    """
    Class to handle products data collecting from ceneo.pl
    """
    def __init__(self, url, product_count, progress_bar_pointer = None):
        """
        Constructor of class CeneoHandler.
        :param string url: the first page of products to be collected
        :param int product_count: the number of products to be collected
        """
        self._main_page_url = url
        self._product_count = product_count
        self._list_of_products = []
        self._product_urls = []
        self._full_data = None

        self._product_urls = self._get_product_urls()
        self._parse_data(progress_bar_pointer)

    def get_collected_data(self):
        """
        Returns the extracted specs data

        :return DataFrame: list of extracted criteria and alternatives
        """
        return self._full_data

    def get_product_urls(self):
        """
        Returns the list of urls for each product

        :return list: product urls
        """
        products = [product[:-len(SPEC_SUFFIX)] for product in self._product_urls]
        return products

    def _get_product_urls(self):
        """
        Method responsible for creating the list of a url links

        :return list: list of strings with url links
        """
        page_counter = 0
        page_count = self._product_count / PRODUCT_PER_PAGE
        page_url = self._main_page_url

        while page_counter < page_count:
            page_soup = request_page_data_soup(page_url)
            self._parse_product_hrefs(page_soup)
            page_url = self._get_next_page_url(page_counter)
            page_counter += 1
            time.sleep(1)

        product_urls = [MAIN_PAGE_LINK + href + SPEC_SUFFIX for href in self._list_of_products]
        return product_urls[:self._product_count]

    def _parse_product_hrefs(self, soup):
        """
        Method updates the list of product urls from given page url.

        :param BeautifulSoup soup: given page soup.
        """
        products = soup.find_all("a",
                                 {"class": "js_clickHash js_seoUrl product-link go-to-product"})
        start_at = 1 if len(products) > PRODUCT_PER_PAGE else 0
        self._list_of_products = [product["href"] for product in products[start_at:]]

    def _parse_data(self, progress_bar_pointer=None):
        """
        Method for parsing the specs data from given urls. It also updates the progress bar in GUI.
        :param list product_urls: list of strings with url links
        :param QProgressBar progress_bar_pointer: pointer to a QProgressBar to show the progress of
                                                  collected data from urls.
        """
        all_collected_specs_df = pd.DataFrame()
        product_counter = 0
        for url in self._product_urls:
            page_soup = request_page_data_soup(url)
            price = page_soup.find("span", {"class": "price-format nowrap"})
            df = pd.DataFrame([{'Cena': price.text}])

            for tr in page_soup.find_all("tr"):
                if len(tr.find_all("th")) == 0:
                    continue

                values = []
                for th in tr.find_all("th"):
                    category = th.text.split("?", 1)[0].strip()

                if "Popularne" in category:
                    continue

                for li in tr.find_all("li"):
                    value = li.text.split(">", 1)[0].strip()
                    values.append(value)

                if len(tr.find_all("li")) > 1:
                    for value in values:
                        df.insert(len(df.columns), category + ": " + value, "Supports")
                    continue

                df.insert(len(df.columns), category, value)

            if all_collected_specs_df.empty:
                all_collected_specs_df = df

            else:
                all_collected_specs_df = pd.concat(
                    [all_collected_specs_df, df],
                    axis=0,
                    join="outer",
                    ignore_index=False,
                    keys=None,
                    levels=None,
                    names=None,
                    verify_integrity=False,
                    copy=True,
                )
            product_counter += 1
            if progress_bar_pointer:
                time.sleep(0.2)
                progress_bar_pointer.setValue(int((product_counter / self._product_count) *
                                                  COMPLETE_PROGRESS_BAR))

        if progress_bar_pointer:
            progress_bar_pointer.setValue(COMPLETE_PROGRESS_BAR)
        all_collected_specs_df.reset_index(drop=True, inplace=True)

        all_collected_specs_df.fillna("Does not support", inplace=True)
        all_collected_specs_df.replace(np.nan, "Does not support")
        all_collected_specs_df.replace(np.NaN, "Does not support")
        self._full_data = all_collected_specs_df

    def _get_next_page_url(self, page_counter):
        """
        Method to prepare next page url address.
        :param int page_counter: indicates the number of the actual page

        :return string: page url to a next page
        """
        next_page_url = self._main_page_url + CENEO_NEXT_PAGE_STYLE + str(page_counter) + ";"
        return next_page_url
