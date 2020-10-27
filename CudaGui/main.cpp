#include "mainwindow.h"

#include <QApplication>

#include <QFile>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QFile styleFile( ":/Aqua.qss" );
    styleFile.open( QFile::ReadOnly );

    QString style( styleFile.readAll() );
    a.setStyleSheet( style );

    MainWindow w;
    w.show();
    return a.exec();
}
