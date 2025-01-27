/**
 *Submitted for verification at Etherscan.io on 2017-08-24
*/

pragma solidity ^0.4.0;
contract countGame {

    address public best_gamer;
    uint public count = 0;
    uint public endTime = 1504969200;
    
    function fund() payable {
        require(now <= endTime);
    }
    
    function (){
        require(now<=endTime && count<50);
        best_gamer = msg.sender;
        count++;
    }
    
    function endGame(){
        require(now>endTime || count == 50);
        best_gamer.transfer(this.balance);
    }
    
}