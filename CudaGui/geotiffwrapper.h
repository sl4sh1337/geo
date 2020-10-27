#ifndef GEOTIFFWRAPPER_H
#define GEOTIFFWRAPPER_H
#include "declares.h"
#include <QString>
#include "tiffview.h"
#include <QListWidgetItem>

class GeoTIFFWrapper: public QListWidgetItem
{
    bool edited = false;
public:
    int mods = 0;
    TIFF* tif;
    uint32* tif_data;
    QString filename;
    int tif_data_width, tif_data_height;
    TIFFView* view;
public:
    GeoTIFFWrapper(const QString& fn);
    ~GeoTIFFWrapper();
    void update(uint32 *new_data, int new_width, int new_height);
    void setEdited(bool e);
    bool isEdited() {return edited;}
    void save();
    void save(QString fn);
};

#endif // GEOTIFFWRAPPER_H
