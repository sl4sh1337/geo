#include "multiviewer.h"
#include "ui_multiviewer.h"
#include "mainwindow.h"
#include "tiffview.h"
#include <QDesktopWidget>

MultiViewer::MultiViewer(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::MultiViewer)
{
    ui->setupUi(this);
    setWindowFlags((windowFlags() & ~Qt::WindowContextHelpButtonHint));// | Qt::MSWindowsFixedSizeDialogHint);
    scene = new QGraphicsScene(this);
    ui->graphicsView->setScene(scene);

    move(QApplication::desktop()->width() / 12 + parent->width() + 10, QApplication::desktop()->height() / 12);
}

MultiViewer::~MultiViewer()
{
    delete ui;
}

void MultiViewer::addItem(GeoTIFFWrapper* item)
{
    QImage im((uchar*)item->tif_data, item->tif_data_width, item->tif_data_height, QImage::Format_RGBA8888);
    processed_pictures.emplace_back(item, QPixmap::fromImage(im));

}

void MultiViewer::updateItem(GeoTIFFWrapper *item)
{
    QImage im((uchar*)item->tif_data, item->tif_data_width, item->tif_data_height, QImage::Format_RGBA8888);

    processed_pictures.emplace_back(item, QPixmap::fromImage(im));
}

void MultiViewer::finishAdding()
{
    TIFFView* picture = new TIFFView(processed_pictures.back().second);

    picture->setFlag(QGraphicsItem::ItemIsMovable);
    picture->setFlag(QGraphicsItem::ItemIsSelectable);

    processed_pictures.back().first->view = picture;
    scene->addItem(picture);
    processed_pictures.clear();
}

void MultiViewer::finishUpdating()
{
    for (int i = 0; i < processed_pictures.size(); ++i)
    {
        processed_pictures[i].first->view->setPixmap(processed_pictures[i].second);

    }
    processed_pictures.clear();
}
