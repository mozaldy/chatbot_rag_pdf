from abc import ABC, abstractmethod
from .models import RouteResult

class BaseQueryRouter(ABC):
    @abstractmethod
    async def route(self, query: str) -> RouteResult:
        """
        Analyzes the query and returns the routing result.
        """
        pass
