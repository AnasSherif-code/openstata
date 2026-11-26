"""Data I/O backends for OpenStata (.dta 118/119 and delimited text)."""

from openstata.io.delimited import read_delimited, write_delimited
from openstata.io.dta import read_dta, write_dta

__all__ = ["read_delimited", "read_dta", "write_delimited", "write_dta"]
