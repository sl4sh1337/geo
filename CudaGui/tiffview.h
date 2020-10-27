#ifndef TIFFVIEW_H
#define TIFFVIEW_H
#include <QGraphicsPixmapItem>
#include <QPixmap>

class TIFFView  : public QGraphicsPixmapItem
{
    bool moved = false;
public:
    TIFFView(const QPixmap& map) : QGraphicsPixmapItem(map) {}
    // QGraphicsItem interface
protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *event);

    // QGraphicsItem interface
protected:
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
};

#endif // TIFFVIEW_H
