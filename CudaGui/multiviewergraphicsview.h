#ifndef MULTIVIEWERGRAPHICSVIEW_H
#define MULTIVIEWERGRAPHICSVIEW_H
#include <QGraphicsView>

class MultiviewerGraphicsView : public QGraphicsView
{
    Q_OBJECT
public:
    explicit MultiviewerGraphicsView(QWidget *parent = nullptr) : QGraphicsView(parent) {}


    // QWidget interface

    // QWidget interface
protected:
    void wheelEvent(QWheelEvent *event);
};

#endif // MULTIVIEWERGRAPHICSVIEW_H
