#include "geotiffwrapper.h"
#include <QFileInfo>

GeoTIFFWrapper::GeoTIFFWrapper(const QString& fn) : QListWidgetItem(QFileInfo(fn).fileName(), nullptr, UserType), filename(fn)
{
    this->tif = XTIFFOpen(filename.toStdString().c_str(), "r");
    TIFFGetField(this->tif, TIFFTAG_IMAGEWIDTH, &this->tif_data_width);
    TIFFGetField(this->tif, TIFFTAG_IMAGELENGTH, &this->tif_data_height);
    tif_data = (uint32*)_TIFFmalloc(this->tif_data_width * tif_data_height * sizeof(uint32));
    TIFFReadRGBAImageOriented(tif, this->tif_data_width, this->tif_data_height, tif_data, ORIENTATION_TOPLEFT);
}

GeoTIFFWrapper::~GeoTIFFWrapper()
{
    delete view;
    delete tif_data;
}

void GeoTIFFWrapper::update(uint32 *new_data, int new_width, int new_height)
{
    delete[] this->tif_data;
    tif_data = new_data;
    tif_data_width = new_width;
    tif_data_height = new_height;
}

void GeoTIFFWrapper::setEdited(bool e)
{
    if(e && !edited)
        setText(this->text() + "*");
    else if(!e && edited)
        setText(QFileInfo(filename).fileName());
    edited = e;
}

void GeoTIFFWrapper::save()
{
    save(filename);
}

void GeoTIFFWrapper::save(QString fn)
{
    XTIFFClose(tif);
    QImage im((uchar*)tif_data, tif_data_width, tif_data_height, QImage::Format_RGBA8888);
    im.save(fn);
    setEdited(false);
    mods = 0;

    delete[] tif_data;

    filename = fn;
    this->tif = XTIFFOpen(fn.toStdString().c_str(), "r");
    TIFFGetField(this->tif, TIFFTAG_IMAGEWIDTH, &this->tif_data_width);
    TIFFGetField(this->tif, TIFFTAG_IMAGELENGTH, &this->tif_data_height);
    tif_data = (uint32*)_TIFFmalloc(this->tif_data_width * tif_data_height * sizeof(uint32));
    TIFFReadRGBAImageOriented(tif, this->tif_data_width, this->tif_data_height, tif_data, ORIENTATION_TOPLEFT);

    setText(QFileInfo(fn).fileName());
}
