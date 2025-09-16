import requests
from abc import ABC, abstractmethod
from urllib.parse import urlencode
from database.network.network import DatasetDownloader


class CoinAPIDownloader(DatasetDownloader, ABC):
    def __init__(self, verbose: bool):
        super().__init__(date_column_name=self._get_date_column_name(), verbose=verbose)

        self._api_key_list = [
            '2a236e63-0432-493e-ad41-bc7016331e4c'

        ]

    @property
    def api_key_list(self) -> list[str]:
        return self._api_key_list

    @abstractmethod
    def _get_date_column_name(self) -> str:
        pass

    @abstractmethod
    def _get_request_params(self) -> dict[str, str]:
        pass

    @staticmethod
    def _encode_request_url(base_url: str, request_params: dict, api_key: str) -> str:
        request_params['apikey'] = api_key
        encoded_params = urlencode(request_params)
        return f'{base_url}?{encoded_params}'

    def _get_response(self, base_url: str, request_params: dict) -> requests.Response or None:
        for api_key in self._api_key_list:
            if self._verbose:
                print(f'Using apikey: {api_key}')

            encoded_request_url = self._encode_request_url(
                base_url=base_url,
                request_params=request_params,
                api_key=api_key
            )
            response = requests.get(encoded_request_url)

            if self._verbose:
                print(f'Response Status: {response.status_code} - {response.reason}-{response.text}')

            if response.status_code == 200:
                return response
        return None
