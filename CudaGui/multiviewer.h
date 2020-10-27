#ifndef MULTIVIEWER_H
#define MULTIVIEWER_H

#include <QDialog>
#include "geotiffwrapper.h"

namespace Ui {
class MultiViewer;
}

class MultiViewer : public QDialog
{
    Q_OBJECT

public:
    QGraphicsScene* scene;
    explicit MultiViewer(QWidget *parent = nullptr);
    ~MultiViewer();
    void addItem(GeoTIFFWrapper* item);
    void updateItem(GeoTIFFWrapper* item);
    void finishAdding();
    void finishUpdating();
    std::vector<std::pair<GeoTIFFWrapper*, QPixmap>> processed_pictures;

//private slots:
    //void on_MultiViewer_finished(int result);

private:
    Ui::MultiViewer *ui;
};

#endif // MULTIVIEWER_H
