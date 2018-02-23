package com.SearchEngine.Ranker;

import java.io.FileReader;
import java.io.BufferedReader;
import java.util.ArrayList;
import java.util.HashMap;

public class Ranker {

    /* The number of pages in the graph*/
    private Integer pagesCount;

    /* The Graph adjacency list */
    private HashMap<Integer, ArrayList<Integer>> adjacencyList = new HashMap<Integer, ArrayList<Integer>>();

    /* The out degrees of each page */
    private ArrayList<Integer> outDegrees;

    /* The ranks of the pages */
    private ArrayList<Integer> pagesRank;

    /* The dumping factor */
    private final Double alpha = 0.85;

    /* The maximum number iterations */
    private final Integer maxIterations = 100;

    private void addArc(int from, int to) {
        if (adjacencyList.containsKey(from)) {
            adjacencyList.get(from).add(to);
        } else {
            ArrayList<Integer> newList = new ArrayList<Integer>();
            newList.add(to);
            adjacencyList.put(from, newList);
        }
        outDegrees.set(from, outDegrees.get(from) + 1);
    }

    private void initializeLists(int pagesCount) {
        this.pagesCount = pagesCount;

        outDegrees = new ArrayList<Integer>();
        pagesRank = new ArrayList<Integer>();

        for (int i = 0; i < pagesCount; i++) {
            outDegrees.add(0);
            pagesRank.add(0);
        }
    }

    public void readFile(String filePath) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(filePath));
            String strLine;

            initializeLists(5000);

            // Read file line by line
            while ((strLine = br.readLine()) != null) {

                String[] strs = strLine.trim().split(" ");
                Integer u = Integer.parseInt(strs[0]);
                Integer v = Integer.parseInt(strs[1]);

                // Add arcs
                this.addArc(u, v);

                System.out.println(u.toString() + " " + v.toString());
            }
        } catch (Exception e) {
            //Catch exception if any
            System.err.println("Error: " + e.getMessage());
        }

    }

    public void rankPages() {
        

    }

    // Debugging @Samir55
    private void debugging() {
        System.out.println("The saved edges printing");

        for (Integer vertex : adjacencyList.keySet()) {
            System.out.println("For the vertex " + vertex.toString());
            System.out.println("OutDegrees " + outDegrees.get(vertex).toString());
            for (Integer to : adjacencyList.get(vertex)) {
                System.out.print(to.toString() + " ");
            }
            System.out.println();
        }
    }

}
