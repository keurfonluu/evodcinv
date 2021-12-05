import os


_extension_to_filetype = {}
_reader_map = {}
_writer_map = {}


def register(file_format, extensions, reader, writer=None):
    """
    Register a new input format.

    Parameters
    ----------
    file_format : str
        File format to register.
    extensions : array_like
        List of extensions to associate to the new format.
    reader : callable
        Read fumction.
    writer : callable or None, optional, default None
        Write function.

    """
    for ext in extensions:
        _extension_to_filetype[ext] = file_format

    if reader is not None:
        _reader_map[file_format] = reader

    if writer is not None:
        _writer_map[file_format] = writer


def read(filename, file_format=None, **kwargs):
    """
    Read inversion results.

    Parameters
    ----------
    filename : str
        Input file name.
    file_format : str ('h5', 'json') or None, optional, default None
        Input file format.

    Returns
    -------
    :class:`evodcinv.InversionResult`
        Inversion results.

    """
    if not isinstance(filename, str):
        raise TypeError()

    if file_format is None:
        file_format = filetype_from_filename(filename, _extension_to_filetype)

    else:
        if file_format not in _reader_map:
            raise ValueError()

    return _reader_map[file_format](filename, **kwargs)


def write(filename, result, file_format=None, **kwargs):
    """
    Write TOUGH input file.

    Parameters
    ----------
    filename : str
        Output file name.
    result : :class:`evodcinv.InversionResult`
        Inversion results to export.
    file_format : str ('h5', 'json') or None, optional, default None
        Output file format.

    Other Parameters
    ----------------
    compression_opts : int, optional, default 4
        Only if ``file_format = "h5"``. Compression level for gzip compression. May be an integer from 0 to 9.
    indent : int, str or None, optional, default None
        Only if ``file_format = "json"``. Indent level.

    """
    if not isinstance(filename, str):
        raise TypeError()
    
    if file_format is None:
        file_format = filetype_from_filename(filename, _extension_to_filetype)

    else:
        if file_format not in _reader_map:
            raise ValueError()

    _writer_map[file_format](filename, result, **kwargs)


def filetype_from_filename(filename, ext_to_fmt):
    """Determine file type from its extension."""
    ext = os.path.splitext(filename)[1].lower()

    return ext_to_fmt[ext] if ext in ext_to_fmt.keys() else ""
